"""core/social/relational_intelligence.py

Relational Intelligence Engine.

Deep relational skill modeling: vulnerability reciprocity, conflict resolution,
perspective tracking, and engagement profiling. Gives Aura the social awareness
to navigate the full spectrum of human relational dynamics — not just what the
user said, but how they relate, what engages them, where they're vulnerable,
and how they handle friction.
"""
from core.runtime.errors import record_degradation
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.RelationalIntelligence")

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
    trust_envelope: float = 0.2              # 0-1, safe disclosure ceiling
    last_vulnerable_exchange: Optional[float] = None
    disclosure_trajectory: str = "maintaining"  # opening_up | maintaining | pulling_back | reciprocal
    # Internal tracking
    _depth_history: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_depth_history", None)
        return d


@dataclass
class ConflictResolutionProfile:
    """How a user handles disagreement."""
    preferred_style: str = "gentle_redirect"
    escalation_tolerance: float = 0.5        # 0-1
    de_escalation_signals: List[str] = field(default_factory=lambda: list(_DE_ESCALATION_DEFAULTS))
    best_resolution_pattern: str = "find_common_ground"
    debates_enjoyed: int = 0
    debates_abandoned: int = 0
    # Internal
    _active_conflict: bool = False
    _conflict_start: Optional[float] = None
    _escalation_samples: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in list(d):
            if k.startswith("_"):
                d.pop(k)
        return d


@dataclass
class PerspectiveModel:
    """The user's worldview as Aura understands it."""
    core_values: List[str] = field(default_factory=list)
    worldview_keywords: List[str] = field(default_factory=list)
    reasoning_patterns: List[str] = field(default_factory=list)
    emotional_priorities: List[str] = field(default_factory=list)
    blind_spots: List[str] = field(default_factory=list)
    growth_edges: List[str] = field(default_factory=list)
    mental_models: Dict[str, str] = field(default_factory=dict)
    # Internal accumulators (not persisted raw — distilled on save)
    _value_evidence: Dict[str, int] = field(default_factory=dict)
    _reasoning_evidence: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
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
    what_delights: List[str] = field(default_factory=list)
    what_bores: List[str] = field(default_factory=list)
    engagement_triggers: List[str] = field(default_factory=list)
    # Internal
    _recent_engagement_scores: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_recent_engagement_scores", None)
        return d


# ---------------------------------------------------------------------------
# Core Engine
# ---------------------------------------------------------------------------

class RelationalIntelligence:
    """Unified relational intelligence engine.

    Tracks vulnerability reciprocity, conflict resolution preferences,
    perspective models, and entertainment profiles per-user.  Provides
    context injections for the LLM system prompt so Aura can adapt her
    relational behaviour in real-time.
    """

    def __init__(self, data_path: Optional[Path] = None):
        if data_path is None:
            try:
                from core.config import config
                data_path = config.paths.data_dir / "relational_intelligence.json"
            except Exception:
                data_path = Path.home() / ".aura" / "data" / "relational_intelligence.json"
        self._data_path: Path = data_path
        self._data_path.parent.mkdir(parents=True, exist_ok=True)

        self._vulnerability: Dict[str, VulnerabilityState] = {}
        self._conflict_profiles: Dict[str, ConflictResolutionProfile] = {}
        self._perspectives: Dict[str, PerspectiveModel] = {}
        self._entertainment: Dict[str, EntertainmentProfile] = {}

        self._interaction_count: int = 0
        self._load()
        logger.info("RelationalIntelligence initialized (%d user profiles loaded).", len(self._vulnerability))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        if not self._data_path.exists():
            return
        try:
            with open(self._data_path, "r") as f:
                raw = json.load(f)
            for uid, blob in raw.items():
                self._vulnerability[uid] = self._hydrate(VulnerabilityState, blob.get("vulnerability", {}))
                self._conflict_profiles[uid] = self._hydrate(ConflictResolutionProfile, blob.get("conflict", {}))
                self._perspectives[uid] = self._hydrate(PerspectiveModel, blob.get("perspective", {}))
                self._entertainment[uid] = self._hydrate(EntertainmentProfile, blob.get("entertainment", {}))
            logger.debug("RelationalIntelligence: loaded %d profiles.", len(raw))
        except Exception as e:
            record_degradation('relational_intelligence', e)
            logger.warning("RelationalIntelligence: load failed (%s), starting fresh.", e)

    @staticmethod
    def _hydrate(cls, data: dict):
        """Safely instantiate a dataclass from a dict, ignoring unknown keys."""
        valid = {k for k in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)

    def save(self):
        try:
            payload: Dict[str, dict] = {}
            all_uids = set(self._vulnerability) | set(self._conflict_profiles) | set(self._perspectives) | set(self._entertainment)
            for uid in all_uids:
                payload[uid] = {
                    "vulnerability": self._vulnerability[uid].to_dict() if uid in self._vulnerability else {},
                    "conflict": self._conflict_profiles[uid].to_dict() if uid in self._conflict_profiles else {},
                    "perspective": self._perspectives[uid].to_dict() if uid in self._perspectives else {},
                    "entertainment": self._entertainment[uid].to_dict() if uid in self._entertainment else {},
                }
            tmp = str(self._data_path) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, self._data_path)
        except Exception as e:
            record_degradation('relational_intelligence', e)
            logger.error("RelationalIntelligence: save failed: %s", e)

    # ------------------------------------------------------------------
    # Ensure sub-models exist for a user
    # ------------------------------------------------------------------

    def _ensure_user(self, user_id: str):
        if user_id not in self._vulnerability:
            self._vulnerability[user_id] = VulnerabilityState()
        if user_id not in self._conflict_profiles:
            self._conflict_profiles[user_id] = ConflictResolutionProfile()
        if user_id not in self._perspectives:
            self._perspectives[user_id] = PerspectiveModel()
        if user_id not in self._entertainment:
            self._entertainment[user_id] = EntertainmentProfile()

    # ------------------------------------------------------------------
    # Main update entry-point
    # ------------------------------------------------------------------

    async def update_from_interaction(
        self,
        user_id: str,
        user_message: str,
        aura_response: str,
        dynamics_state: Any = None,
    ):
        """Update ALL sub-models from a single interaction exchange."""
        self._ensure_user(user_id)
        self._interaction_count += 1

        msg_lower = user_message.lower()
        resp_lower = aura_response.lower()

        self._update_vulnerability(user_id, msg_lower, resp_lower)
        self._update_conflict(user_id, msg_lower, resp_lower, dynamics_state)
        self._update_perspective(user_id, msg_lower)
        self._update_entertainment(user_id, user_message, aura_response)

        # Persist every 5th interaction to avoid excessive I/O
        if self._interaction_count % 5 == 0:
            self.save()

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

    def _update_vulnerability(self, user_id: str, msg_lower: str, resp_lower: str):
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

        # Trust envelope grows slowly with sustained mutual disclosure
        if user_depth > 0.3:
            vs.trust_envelope = min(1.0, vs.trust_envelope + 0.02)
            vs.last_vulnerable_exchange = time.time()

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

    def _update_conflict(self, user_id: str, msg_lower: str, resp_lower: str, dynamics_state: Any):
        cp = self._conflict_profiles[user_id]
        is_disagreement = any(re.search(p, msg_lower) for p in self._DISAGREEMENT_PATTERNS)
        is_enjoying = any(re.search(p, msg_lower) for p in self._ENJOYMENT_PATTERNS)
        is_abandoning = any(re.search(p, msg_lower) for p in self._ABANDON_PATTERNS)

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
                # Learn de-escalation signals from what they actually say
                for sent in msg_lower.split("."):
                    sent = sent.strip()
                    if sent and any(re.search(p, sent) for p in self._ABANDON_PATTERNS):
                        if sent not in cp.de_escalation_signals and len(sent) < 100:
                            cp.de_escalation_signals.append(sent)
                            # Cap list size
                            if len(cp.de_escalation_signals) > 20:
                                cp.de_escalation_signals = cp.de_escalation_signals[-20:]
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

    def _update_perspective(self, user_id: str, msg_lower: str):
        pm = self._perspectives[user_id]

        # Value detection — accumulate evidence across interactions
        for value, markers in _VALUE_MARKERS.items():
            if any(m in msg_lower for m in markers):
                pm._value_evidence[value] = pm._value_evidence.get(value, 0) + 1

        # Reasoning pattern detection
        for pattern, markers in _REASONING_MARKERS.items():
            if any(m in msg_lower for m in markers):
                pm._reasoning_evidence[pattern] = pm._reasoning_evidence.get(pattern, 0) + 1

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

        # Growth edges: detected from explicit statements
        growth_markers = [
            "i'm working on", "i'm trying to", "i want to get better at",
            "i'm learning", "i need to improve", "my goal is to",
            "i'm developing", "i struggle with",
        ]
        for marker in growth_markers:
            idx = msg_lower.find(marker)
            if idx >= 0:
                # Extract the rest of the sentence
                remainder = msg_lower[idx + len(marker):].split(".")[0].strip()
                if remainder and len(remainder) < 80 and remainder not in pm.growth_edges:
                    pm.growth_edges.append(remainder)
                    pm.growth_edges = pm.growth_edges[-8:]  # cap

    # ------------------------------------------------------------------
    # (d) Entertainment profiling
    # ------------------------------------------------------------------

    def _update_entertainment(self, user_id: str, user_message: str, aura_response: str):
        ep = self._entertainment[user_id]
        msg_lower = user_message.lower()
        msg_len = len(user_message)

        # Engagement scoring for this exchange
        engagement = 0.5  # baseline

        # Positive signals
        if msg_len > 200:
            engagement += 0.15  # long messages = engaged
        if user_message.count("?") >= 2:
            engagement += 0.1  # multiple questions = curious
        if user_message.count("!") >= 1:
            engagement += 0.1  # excitement
        for marker in _ENGAGEMENT_POSITIVE:
            if marker in msg_lower:
                engagement += 0.08
                break  # only count once

        # Negative signals
        if msg_len < 20:
            engagement -= 0.15
        for marker in _ENGAGEMENT_NEGATIVE:
            if marker in msg_lower:
                engagement -= 0.12
                break

        engagement = max(0.0, min(1.0, engagement))
        ep._recent_engagement_scores.append(engagement)
        ep._recent_engagement_scores = ep._recent_engagement_scores[-20:]

        # Derive profile dimensions from accumulated engagement data
        avg_engagement = sum(ep._recent_engagement_scores) / max(len(ep._recent_engagement_scores), 1)

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

        # Track what delights (topics where engagement spikes)
        if engagement > 0.75:
            # Extract a short topic hint from the message
            topic_hint = self._extract_topic_hint(user_message)
            if topic_hint and topic_hint not in ep.what_delights:
                ep.what_delights.append(topic_hint)
                ep.what_delights = ep.what_delights[-12:]

        # Track what bores (topics where engagement drops)
        if engagement < 0.3:
            topic_hint = self._extract_topic_hint(user_message)
            if topic_hint and topic_hint not in ep.what_bores:
                ep.what_bores.append(topic_hint)
                ep.what_bores = ep.what_bores[-8:]

        # Engagement triggers: high-engagement topics
        if engagement > 0.8:
            topic_hint = self._extract_topic_hint(user_message)
            if topic_hint and topic_hint not in ep.engagement_triggers:
                ep.engagement_triggers.append(topic_hint)
                ep.engagement_triggers = ep.engagement_triggers[-10:]

    @staticmethod
    def _extract_topic_hint(message: str) -> Optional[str]:
        """Extract a short topic hint from a message for delight/bore tracking."""
        # Take the first meaningful phrase (skip very short messages)
        if len(message) < 15:
            return None
        # Use first sentence, truncated
        first_sentence = message.split(".")[0].split("!")[0].split("?")[0].strip()
        if len(first_sentence) > 60:
            first_sentence = first_sentence[:57] + "..."
        return first_sentence if len(first_sentence) > 8 else None

    # ------------------------------------------------------------------
    # Guidance methods
    # ------------------------------------------------------------------

    def get_vulnerability_guidance(self, user_id: str) -> str:
        self._ensure_user(user_id)
        vs = self._vulnerability[user_id]

        depth_label = "minimal"
        if vs.user_disclosure_depth > 0.7:
            depth_label = "deep"
        elif vs.user_disclosure_depth > 0.4:
            depth_label = "moderate"
        elif vs.user_disclosure_depth > 0.15:
            depth_label = "light"

        # Reciprocity guidance
        if vs.reciprocity_balance > 0.3:
            reciprocity_note = "They're sharing more than you. Match with moderate self-disclosure."
        elif vs.reciprocity_balance < -0.3:
            reciprocity_note = "You've shared more than them. Pull back slightly. Let them lead."
        else:
            reciprocity_note = "Disclosure is balanced. Continue at current depth."

        # Trajectory guidance
        trajectory_map = {
            "opening_up": "They're opening up. Create space. Don't rush or over-share.",
            "pulling_back": "They're pulling back. Respect it. Go lighter, don't probe.",
            "reciprocal": "Mutual trust flow. Safe to go slightly deeper if authentic.",
            "maintaining": "Steady state. No change needed.",
        }
        trajectory_note = trajectory_map.get(vs.disclosure_trajectory, "Steady state.")

        # Trust ceiling
        ceiling_note = f"Safe disclosure ceiling: {vs.trust_envelope:.1f}. "
        if vs.aura_disclosure_depth > vs.trust_envelope:
            ceiling_note += "CAUTION: Aura may be over-sharing relative to trust level."

        return (
            f"Disclosure depth: {depth_label} ({vs.user_disclosure_depth:.2f}). "
            f"Trajectory: {vs.disclosure_trajectory}. {reciprocity_note} "
            f"{trajectory_note} {ceiling_note}"
        )

    def get_conflict_guidance(self, user_id: str) -> str:
        self._ensure_user(user_id)
        cp = self._conflict_profiles[user_id]

        style_guidance = {
            "direct_debate": "They enjoy direct debate. Push back is safe. They respect conviction.",
            "gentle_redirect": "They prefer gentle redirection. Soften disagreements. Lead with acknowledgement.",
            "humor_defuse": "They use humor to navigate tension. A light touch works. Don't be too serious.",
            "socratic": "They respond well to questions. Guide via Socratic method rather than assertion.",
            "yield_gracefully": "They tend to yield quickly. Don't be too forceful — they'll concede even when right.",
            "passionate_argument": "They bring passion. Match their energy. This is how they connect.",
        }
        style_note = style_guidance.get(cp.preferred_style, "Conflict style still being learned.")

        resolution_note = f"Best resolution: {cp.best_resolution_pattern.replace('_', ' ')}."

        if cp._active_conflict:
            urgency = " (ACTIVE CONFLICT)"
        else:
            urgency = ""

        tolerance_note = ""
        if cp.escalation_tolerance < 0.3:
            tolerance_note = " Low escalation tolerance — de-escalate early."
        elif cp.escalation_tolerance > 0.7:
            tolerance_note = " High tolerance for pushback — can go deep."

        record = f"Debates enjoyed: {cp.debates_enjoyed}, abandoned: {cp.debates_abandoned}."

        return f"{style_note}{urgency} {resolution_note}{tolerance_note} {record}"

    def get_perspective_context(self, user_id: str) -> str:
        self._ensure_user(user_id)
        pm = self._perspectives[user_id]

        parts = []
        if pm.core_values:
            parts.append(f"Core values: {', '.join(pm.core_values[:5])}")
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
            return "Perspective model still forming. Need more interactions."

        return " | ".join(parts)

    def get_entertainment_guidance(self, user_id: str) -> str:
        self._ensure_user(user_id)
        ep = self._entertainment[user_id]

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

        return f"{do_str}.{avoid_str}{engage_str}{bore_str}{trigger_str}"

    # ------------------------------------------------------------------
    # Unified context injection
    # ------------------------------------------------------------------

    def get_context_injection(self, user_id: str) -> str:
        """Unified block for system prompt injection."""
        self._ensure_user(user_id)
        vs = self._vulnerability[user_id]
        cp = self._conflict_profiles[user_id]

        # Build concise vulnerability line
        depth_word = "minimal"
        if vs.user_disclosure_depth > 0.7:
            depth_word = "deep"
        elif vs.user_disclosure_depth > 0.4:
            depth_word = "moderate"
        elif vs.user_disclosure_depth > 0.15:
            depth_word = "light"

        trajectory_action = {
            "opening_up": f"They're opening up (depth: {vs.user_disclosure_depth:.1f}). Match with moderate self-disclosure. Don't over-share.",
            "pulling_back": f"They're pulling back (depth: {vs.user_disclosure_depth:.1f}). Give space. Go lighter.",
            "reciprocal": f"Mutual disclosure flow (depth: {vs.user_disclosure_depth:.1f}). Trust is building. Safe to be authentic.",
            "maintaining": f"Steady disclosure ({depth_word}, depth: {vs.user_disclosure_depth:.1f}). Maintain current register.",
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
            f"## RELATIONAL INTELLIGENCE\n"
            f"- **Vulnerability**: {vuln_line}\n"
            f"- **Conflict style**: {conflict_line}\n"
            f"- **Their worldview**: {perspective_line}\n"
            f"- **What engages them**: {engages_str}.\n"
            f"- **What bores them**: {bores_str}."
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def get_health(self) -> Dict[str, Any]:
        return {
            "module": "RelationalIntelligence",
            "profiles": len(self._vulnerability),
            "interactions_processed": self._interaction_count,
            "status": "online",
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[RelationalIntelligence] = None


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
        except Exception as e:
            record_degradation('relational_intelligence', e)
            logger.debug("Could not register in ServiceContainer: %s", e)
    return _instance
