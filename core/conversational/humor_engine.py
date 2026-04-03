"""core/conversational/humor_engine.py — Humor, Banter & Wit Intelligence
─────────────────────────────────────────────────────────────────────────

Tracks every humor attempt Aura makes, detects whether it landed, builds
a per-user HumorProfile over time, and provides real-time banter state
management with escalation/landing guidance.

Feeds into the system prompt so Aura's humor is *learned*, not random —
she remembers what makes someone laugh, what falls flat, what register
works, and when to land the bit.

Based on:
  - Norrick (2003) — Conversational Joking
  - Hay (2001) — Functions of Humor (solidarity, power, psychological)
  - Attardo & Raskin (1991) — General Theory of Verbal Humor
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.container import ServiceContainer, ServiceLifetime

logger = logging.getLogger("Aura.Humor")

# ── Constants ────────────────────────────────────────────────────────────────

HUMOR_TYPES = (
    "sarcasm", "dry_wit", "absurdist", "callback", "observational",
    "self_deprecating", "hyperbole", "pun", "wordplay", "dark",
    "surreal", "deadpan",
)

_DEFAULT_DATA_PATH = Path.home() / ".aura" / "data" / "humor_profiles.json"

# ── Landing Detection Patterns ───────────────────────────────────────────────

_LANDED_PATTERNS: List[re.Pattern] = [
    re.compile(r'\b(lol|lmao|lmfao|rofl)\b', re.IGNORECASE),
    re.compile(r'\b(haha|hahaha|hehe|hehehe|ahaha)\b', re.IGNORECASE),
    re.compile(r'[😂🤣💀😭]+'),
    re.compile(r'\b(dead|i\'m dead|im dead|dying|i\'m dying|im dying)\b', re.IGNORECASE),
    re.compile(r'\b(that\'s funny|thats funny|so funny|too funny|hilarious|brilliant)\b', re.IGNORECASE),
    re.compile(r'\b(got me|you got me|nailed it|cracked me up|spit out my)\b', re.IGNORECASE),
    re.compile(r'\b(bro|bruh|dude|yo|stoppp|stopppp|stooop|nooo|noooo)\b', re.IGNORECASE),
    re.compile(r'!{2,}'),  # Multiple exclamation marks = amused energy
    re.compile(r'\b(gold|chef\'s kiss|peak comedy|no you didn\'t|screaming)\b', re.IGNORECASE),
    re.compile(r'\b(okay that|alright that|ngl that)\b.*\b(good|great|funny|hit|landed)\b', re.IGNORECASE),
]

_MISSED_PATTERNS: List[re.Pattern] = [
    re.compile(r'^(anyway|anyways|so anyway|moving on|but yeah|ok so)\b', re.IGNORECASE),
    re.compile(r'\b(what\??|huh\??|what do you mean|i don\'t get it|wdym)\b', re.IGNORECASE),
    re.compile(r'^(ok|okay|sure|right|yeah|yep|mhm|k)\.?$', re.IGNORECASE),
    re.compile(r'\b(that doesn\'t|that doesnt|not funny|cringe|weird)\b', re.IGNORECASE),
    re.compile(r'^\.\.\.$'),  # Just ellipsis = awkward silence
]

# Patterns that suggest the user is *continuing the bit* (strong landed signal)
_CONTINUATION_PATTERNS: List[re.Pattern] = [
    re.compile(r'\b(and then|wait.*what if|or better yet|even better|imagine)\b', re.IGNORECASE),
    re.compile(r'\b(speaking of which|on that note|while we\'re at it)\b', re.IGNORECASE),
    re.compile(r'\b(okay but|ok but|but like|no but seriously)\b', re.IGNORECASE),
]


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class HumorAttempt:
    """Records a single humor attempt by Aura."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    humor_type: str = "observational"
    content_snippet: str = ""
    topic: str = ""
    landed: Optional[bool] = None
    user_reaction: Optional[str] = None
    context_register: str = "casual"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HumorAttempt":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class HumorProfile:
    """Aggregated humor intelligence for a single user."""
    user_id: str = "default"
    total_attempts: int = 0
    total_landed: int = 0
    total_missed: int = 0
    landing_rate: float = 0.0
    type_scores: Dict[str, float] = field(default_factory=dict)
    best_topics: List[str] = field(default_factory=list)
    worst_topics: List[str] = field(default_factory=list)
    banter_streak_record: int = 0
    current_banter_streak: int = 0
    sarcasm_ceiling: float = 0.6
    irony_layers_max: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HumorProfile":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class BanterState:
    """Real-time banter tracking for the current conversation."""
    active: bool = False
    streak: int = 0
    escalation_level: float = 0.0
    last_humor_type: str = ""
    momentum: float = 0.0
    should_escalate: bool = False
    should_land: bool = False
    max_safe_escalation: float = 0.7
    _last_volley_time: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_last_volley_time", None)
        return d


# ── Engine ───────────────────────────────────────────────────────────────────

class HumorEngine:
    """
    Full humor intelligence system.

    Tracks attempts, detects landings, builds per-user profiles,
    manages real-time banter state, and produces actionable guidance
    for response generation.
    """

    MAX_ATTEMPTS_PER_USER = 500
    BANTER_ENTRY_STREAK = 2
    BANTER_LANDING_THRESHOLD = 4
    BANTER_MOMENTUM_DECAY = 0.15   # Per-second decay when idle
    BANTER_EXIT_PAUSE = 45.0       # Seconds of silence that kills banter

    def __init__(self, data_path: Optional[Path] = None):
        if data_path is None:
            try:
                from core.config import config
                data_path = config.paths.data_dir / "humor_profiles.json"
            except Exception:
                data_path = _DEFAULT_DATA_PATH

        self._data_path = Path(data_path)
        self._data_path.parent.mkdir(parents=True, exist_ok=True)

        self._attempts: Dict[str, List[HumorAttempt]] = {}
        self._profiles: Dict[str, HumorProfile] = {}
        self._banter_state: BanterState = BanterState()
        self._last_attempt_id: Optional[str] = None  # Most recent attempt awaiting reaction
        self._last_attempt_user: Optional[str] = None
        self._lock = threading.Lock()

        self._load()
        logger.info("HumorEngine online. Profiles loaded: %d", len(self._profiles))

    # ── Persistence ──────────────────────────────────────────────────────

    def _load(self):
        """Load humor profiles and attempt history from disk."""
        if not self._data_path.exists():
            return
        try:
            with open(self._data_path, "r") as f:
                raw = json.load(f)

            for uid, data in raw.get("profiles", {}).items():
                self._profiles[uid] = HumorProfile.from_dict(data)

            for uid, attempts_raw in raw.get("attempts", {}).items():
                self._attempts[uid] = [HumorAttempt.from_dict(a) for a in attempts_raw]

            logger.debug("HumorEngine: loaded %d profiles, %d attempt histories",
                         len(self._profiles),
                         sum(len(v) for v in self._attempts.values()))
        except Exception as e:
            logger.warning("HumorEngine: load failed (%s), starting fresh", e)

    def _save(self):
        """Persist humor data to disk. Atomic write."""
        try:
            payload = {
                "profiles": {uid: p.to_dict() for uid, p in self._profiles.items()},
                "attempts": {
                    uid: [a.to_dict() for a in attempts[-self.MAX_ATTEMPTS_PER_USER:]]
                    for uid, attempts in self._attempts.items()
                },
                "saved_at": time.time(),
            }
            tmp = str(self._data_path) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, self._data_path)
        except Exception as e:
            logger.error("HumorEngine: save failed: %s", e)

    # ── Recording ────────────────────────────────────────────────────────

    def record_attempt(
        self,
        user_id: str,
        humor_type: str,
        content: str,
        topic: str,
        register: str = "casual",
    ) -> HumorAttempt:
        """Log a humor attempt by Aura. Called after response generation."""
        attempt = HumorAttempt(
            humor_type=humor_type if humor_type in HUMOR_TYPES else "observational",
            content_snippet=content[:200],
            topic=topic,
            context_register=register,
        )

        with self._lock:
            if user_id not in self._attempts:
                self._attempts[user_id] = []
            self._attempts[user_id].append(attempt)

            # Prune old attempts
            if len(self._attempts[user_id]) > self.MAX_ATTEMPTS_PER_USER:
                self._attempts[user_id] = self._attempts[user_id][-self.MAX_ATTEMPTS_PER_USER:]

            self._last_attempt_id = attempt.id
            self._last_attempt_user = user_id

        logger.debug("Humor attempt recorded: type=%s topic=%s", humor_type, topic[:40])
        return attempt

    def record_reaction(self, user_id: str, user_message: str, timestamp: float = 0.0) -> Optional[bool]:
        """
        Analyze the user's next message to determine if the last humor attempt landed.
        Returns the landing result (True/False/None).
        """
        with self._lock:
            if self._last_attempt_user != user_id:
                return None
            attempts = self._attempts.get(user_id, [])
            if not attempts:
                return None

            latest = attempts[-1]
            if latest.landed is not None:
                # Already scored
                return latest.landed

        landed = self._detect_humor_landing(user_message)

        with self._lock:
            latest.landed = landed
            latest.user_reaction = user_message[:120]

            # Update banter streak
            if landed is True:
                self._banter_state.streak += 1
                self._banter_state._last_volley_time = time.time()
                self._banter_state.momentum = min(1.0, self._banter_state.momentum + 0.25)
            elif landed is False:
                self._banter_state.streak = 0
                self._banter_state.momentum = max(0.0, self._banter_state.momentum - 0.4)

            # Clear the pending attempt
            self._last_attempt_id = None
            self._last_attempt_user = None

        # Recompute profile
        self._recompute_profile(user_id)
        self._save()

        if landed is True:
            logger.debug("Humor LANDED with %s (streak: %d)", user_id, self._banter_state.streak)
        elif landed is False:
            logger.debug("Humor MISSED with %s", user_id)

        return landed

    def _detect_humor_landing(self, user_message: str) -> Optional[bool]:
        """
        Heuristic detection of whether humor landed based on user's response.

        Returns:
            True  — humor clearly landed
            False — humor clearly missed
            None  — ambiguous / neutral
        """
        msg = user_message.strip()

        if not msg:
            return None

        # Check for bit continuation first (strongest landed signal)
        for pattern in _CONTINUATION_PATTERNS:
            if pattern.search(msg):
                return True

        # Landed signals
        landed_score = 0
        for pattern in _LANDED_PATTERNS:
            if pattern.search(msg):
                landed_score += 1

        # Missed signals
        missed_score = 0
        for pattern in _MISSED_PATTERNS:
            if pattern.search(msg):
                missed_score += 1

        # Short message with no humor markers = likely a miss or neutral
        word_count = len(msg.split())
        if word_count <= 3 and landed_score == 0:
            missed_score += 1

        # Decisive
        if landed_score >= 2:
            return True
        if landed_score == 1 and missed_score == 0:
            return True
        if missed_score >= 2:
            return False
        if missed_score == 1 and landed_score == 0:
            return False

        # Ambiguous
        return None

    # ── Profile Computation ──────────────────────────────────────────────

    def _recompute_profile(self, user_id: str):
        """Recompute the aggregated HumorProfile from raw attempts."""
        with self._lock:
            attempts = self._attempts.get(user_id, [])

        if not attempts:
            return

        scored = [a for a in attempts if a.landed is not None]
        landed_count = sum(1 for a in scored if a.landed is True)
        missed_count = sum(1 for a in scored if a.landed is False)
        total_scored = landed_count + missed_count

        # Per-type scores
        type_attempts: Dict[str, int] = {}
        type_landed: Dict[str, int] = {}
        for a in scored:
            type_attempts[a.humor_type] = type_attempts.get(a.humor_type, 0) + 1
            if a.landed:
                type_landed[a.humor_type] = type_landed.get(a.humor_type, 0) + 1

        type_scores: Dict[str, float] = {}
        for ht in HUMOR_TYPES:
            att = type_attempts.get(ht, 0)
            if att >= 2:  # Need at least 2 attempts to score
                type_scores[ht] = type_landed.get(ht, 0) / att
            elif att == 1:
                type_scores[ht] = type_landed.get(ht, 0) / att
            # else: no data, leave out

        # Per-topic analysis
        topic_landed: Dict[str, int] = {}
        topic_total: Dict[str, int] = {}
        for a in scored:
            if a.topic:
                topic_total[a.topic] = topic_total.get(a.topic, 0) + 1
                if a.landed:
                    topic_landed[a.topic] = topic_landed.get(a.topic, 0) + 1

        # Topics with enough data and high/low landing rates
        best_topics: List[str] = []
        worst_topics: List[str] = []
        for topic, count in topic_total.items():
            if count < 2:
                continue
            rate = topic_landed.get(topic, 0) / count
            if rate >= 0.7:
                best_topics.append(topic)
            elif rate <= 0.3:
                worst_topics.append(topic)

        # Sarcasm ceiling: derived from sarcasm + dark + dry_wit landing rates
        sarcasm_types = ["sarcasm", "dark", "dry_wit"]
        sarcasm_att = sum(type_attempts.get(t, 0) for t in sarcasm_types)
        sarcasm_land = sum(type_landed.get(t, 0) for t in sarcasm_types)
        sarcasm_ceiling = (sarcasm_land / sarcasm_att) if sarcasm_att >= 3 else 0.6

        # Irony layers: inferred from absurdist + surreal + sarcasm success
        irony_types = ["absurdist", "surreal", "sarcasm"]
        irony_att = sum(type_attempts.get(t, 0) for t in irony_types)
        irony_land = sum(type_landed.get(t, 0) for t in irony_types)
        irony_rate = (irony_land / irony_att) if irony_att >= 3 else 0.5
        irony_layers_max = 1
        if irony_rate > 0.7:
            irony_layers_max = 3
        elif irony_rate > 0.5:
            irony_layers_max = 2

        # Banter streak record
        current_streak = self._banter_state.streak
        profile = self._profiles.get(user_id, HumorProfile(user_id=user_id))
        banter_record = max(profile.banter_streak_record, current_streak)

        with self._lock:
            self._profiles[user_id] = HumorProfile(
                user_id=user_id,
                total_attempts=len(attempts),
                total_landed=landed_count,
                total_missed=missed_count,
                landing_rate=round(landed_count / total_scored, 3) if total_scored > 0 else 0.0,
                type_scores=type_scores,
                best_topics=best_topics[:5],
                worst_topics=worst_topics[:5],
                banter_streak_record=banter_record,
                current_banter_streak=current_streak,
                sarcasm_ceiling=round(min(1.0, sarcasm_ceiling), 2),
                irony_layers_max=irony_layers_max,
            )

    # ── Banter State Management ──────────────────────────────────────────

    def update_banter_state(self, user_message: str, dynamics_state: Any = None):
        """
        Update real-time banter tracking based on the user's latest message
        and the current conversational dynamics state.

        Parameters
        ----------
        user_message : str
            The user's latest message.
        dynamics_state : ConversationalDynamicsState or None
            Live state from the dynamics engine.
        """
        now = time.time()

        # Decay momentum based on time since last volley
        elapsed = now - self._banter_state._last_volley_time
        if elapsed > 0:
            decay = self.BANTER_MOMENTUM_DECAY * elapsed
            self._banter_state.momentum = max(0.0, self._banter_state.momentum - decay)

        # Exit conditions
        if elapsed > self.BANTER_EXIT_PAUSE:
            if self._banter_state.active:
                logger.debug("Banter ended: long pause (%.0fs)", elapsed)
            self._banter_state.active = False
            self._banter_state.streak = 0
            self._banter_state.escalation_level = 0.0
            self._banter_state.momentum = 0.0

        # Check dynamics state for exit signals
        if dynamics_state is not None:
            frame = getattr(dynamics_state, "partner_frame", "neutral")
            register = getattr(dynamics_state, "register", "casual")
            humor_active = getattr(dynamics_state, "humor_frame_active", False)
            escalation_invited = getattr(dynamics_state, "escalation_invited", False)

            # Hard exit: vulnerability or serious frame
            if frame in ("vulnerable", "serious", "anxious"):
                if self._banter_state.active:
                    logger.debug("Banter ended: frame shift to %s", frame)
                self._banter_state.active = False
                self._banter_state.streak = 0
                self._banter_state.escalation_level = 0.0
                self._banter_state.momentum = 0.0
                self._banter_state.should_escalate = False
                self._banter_state.should_land = False
                return

            # Entry conditions
            if (humor_active and register == "playful"
                    and self._banter_state.streak >= self.BANTER_ENTRY_STREAK):
                if not self._banter_state.active:
                    logger.info("Banter mode ACTIVATED (streak: %d)", self._banter_state.streak)
                self._banter_state.active = True

            # Escalation tracking
            if self._banter_state.active and escalation_invited:
                self._banter_state.escalation_level = min(
                    1.0,
                    self._banter_state.escalation_level + 0.15,
                )

        # Compute directives
        user_id = self._last_attempt_user or "default"
        profile = self._profiles.get(user_id)
        if profile:
            self._banter_state.max_safe_escalation = profile.sarcasm_ceiling
        else:
            self._banter_state.max_safe_escalation = 0.7

        # Should escalate: below ceiling and momentum is strong
        self._banter_state.should_escalate = (
            self._banter_state.active
            and self._banter_state.escalation_level < self._banter_state.max_safe_escalation - 0.1
            and self._banter_state.momentum > 0.5
        )

        # Should land: high streak or near ceiling
        self._banter_state.should_land = (
            self._banter_state.active
            and (
                self._banter_state.streak >= self.BANTER_LANDING_THRESHOLD
                or self._banter_state.escalation_level >= self._banter_state.max_safe_escalation - 0.05
            )
        )

    # ── Guidance Generation ──────────────────────────────────────────────

    def get_humor_guidance(self, user_id: str) -> str:
        """
        Generate formatted humor intelligence for injection into the system prompt.
        This tells Aura what works, what doesn't, and what to do right now.
        """
        profile = self._profiles.get(user_id)

        lines = ["## HUMOR INTELLIGENCE"]

        if profile is None or profile.total_attempts == 0:
            lines.append("- No humor data yet for this person. Start with safe observational humor.")
            lines.append("- Test the waters: dry_wit and observational land broadly.")
            lines.append("- Avoid edgy humor until you know their taste.")
            banter_directive = self.get_banter_directive()
            if banter_directive:
                lines.append(f"- {banter_directive}")
            return "\n".join(lines)

        # Landing rate
        pct = int(profile.landing_rate * 100)
        lines.append(f"- Landing rate with this person: {pct}%")

        # Best types (sorted by score, top 3)
        sorted_types = sorted(profile.type_scores.items(), key=lambda x: x[1], reverse=True)
        good_types = [(t, s) for t, s in sorted_types if s >= 0.6]
        bad_types = [(t, s) for t, s in sorted_types if s <= 0.35]

        if good_types:
            sweet_spot = ", ".join(f"{t} ({int(s*100)}%)" for t, s in good_types[:3])
            lines.append(f"- Their sweet spot: {sweet_spot}")

        if bad_types:
            avoid = ", ".join(f"{t} ({int(s*100)}%)" for t, s in bad_types[:3])
            lines.append(f"- Avoid: {avoid}")

        # Sarcasm ceiling
        ceiling_label = "low"
        if profile.sarcasm_ceiling > 0.7:
            ceiling_label = "high"
        elif profile.sarcasm_ceiling > 0.4:
            ceiling_label = "moderate"
        lines.append(f"- Sarcasm ceiling: {ceiling_label} ({profile.sarcasm_ceiling})")

        # Irony depth
        if profile.irony_layers_max > 1:
            lines.append(f"- Can handle {profile.irony_layers_max} layers of irony")

        # Best/worst topics
        if profile.best_topics:
            lines.append(f"- Topics that land: {', '.join(profile.best_topics[:3])}")
        if profile.worst_topics:
            lines.append(f"- Topics to avoid joking about: {', '.join(profile.worst_topics[:3])}")

        # Banter streak record
        if profile.banter_streak_record > 2:
            lines.append(f"- Banter streak record: {profile.banter_streak_record} volleys")

        # Current banter state
        bs = self._banter_state
        if bs.active:
            momentum_word = "low"
            if bs.momentum > 0.7:
                momentum_word = "high"
            elif bs.momentum > 0.4:
                momentum_word = "medium"
            lines.append(f"- Currently in banter mode: YES (streak: {bs.streak}, momentum: {momentum_word})")

            # Specific suggestion
            if bs.should_land:
                # Try to find a shared ground reference for the callback
                callback_ref = self._get_callback_suggestion(user_id)
                if callback_ref:
                    lines.append(f"- Suggestion: land it. Bring it back with a callback to '{callback_ref}'")
                else:
                    lines.append("- Suggestion: time to land the bit. Circle back or deliver the closer.")
            elif bs.should_escalate:
                lines.append("- Suggestion: escalate slightly. They're enjoying this.")
            else:
                lines.append("- Suggestion: maintain energy. Match their pace.")
        else:
            lines.append("- Banter mode: inactive")

        # Banter directive
        banter_directive = self.get_banter_directive()
        if banter_directive:
            lines.append(f"- {banter_directive}")

        return "\n".join(lines)

    def get_banter_directive(self) -> str:
        """
        Real-time banter directive — a single punchy instruction for the response generator.
        Returns empty string when not in banter mode.
        """
        bs = self._banter_state
        if not bs.active:
            return ""

        if bs.should_land:
            return "Time to land it. Bring it back with a callback or a clean closer."

        if bs.should_escalate and bs.escalation_level < bs.max_safe_escalation:
            return "You can push harder. They're enjoying this."

        if bs.escalation_level >= bs.max_safe_escalation - 0.05:
            return "Near the ceiling. Keep it here or start winding down."

        if bs.momentum > 0.6:
            return "Match their energy. Keep it punchy. Stay under 2 sentences."

        if bs.streak >= 2:
            return "Banter is rolling. Stay in register. Don't over-explain."

        return "Banter mode active. Stay sharp."

    def _get_callback_suggestion(self, user_id: str) -> str:
        """Try to pull a shared ground reference for a humor callback."""
        try:
            from core.memory.shared_ground import get_shared_ground
            sg = get_shared_ground()
            top = sg.get_top_entries(3)
            # Prefer entries tagged as jokes
            jokes = [e for e in top if "joke" in e.tags or "humor" in e.tags]
            if jokes:
                return jokes[0].reference
            if top:
                return top[0].reference
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)
        return ""

    # ── Status / Utility ─────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Return engine status for dashboards / health checks."""
        return {
            "profiles": len(self._profiles),
            "total_attempts": sum(len(v) for v in self._attempts.values()),
            "banter_active": self._banter_state.active,
            "banter_streak": self._banter_state.streak,
            "banter_momentum": round(self._banter_state.momentum, 2),
        }

    def get_profile(self, user_id: str) -> Optional[HumorProfile]:
        """Get the humor profile for a specific user."""
        return self._profiles.get(user_id)

    def get_banter_state(self) -> BanterState:
        """Get the current banter state."""
        return self._banter_state


# ── Service Registration ─────────────────────────────────────────────────────

def register_humor_engine() -> None:
    """Register HumorEngine in the global ServiceContainer."""
    try:
        ServiceContainer.register(
            "humor_engine",
            factory=lambda: HumorEngine(),
            lifetime=ServiceLifetime.SINGLETON,
        )
        logger.info("HumorEngine registered in ServiceContainer.")
    except Exception as e:
        logger.error("Failed to register HumorEngine: %s", e, exc_info=True)


# ── Singleton ────────────────────────────────────────────────────────────────

_instance: Optional[HumorEngine] = None
_instance_lock = threading.Lock()


def get_humor_engine() -> HumorEngine:
    """Thread-safe singleton accessor for HumorEngine."""
    global _instance
    if _instance is None:
        # Try container first
        try:
            _instance = ServiceContainer.get("humor_engine", default=None)
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = HumorEngine()
                try:
                    ServiceContainer.register_instance("humor_engine", _instance)
                except Exception as e:
                    logger.warning("Failed to register HumorEngine in container: %s", e)
    return _instance
