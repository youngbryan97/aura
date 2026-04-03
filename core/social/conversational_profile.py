"""core/social/conversational_profile.py
Persistent per-user conversational intelligence profile.

Builds a detailed model of HOW each user communicates: humor preferences,
vulnerability pace, linguistic signature, intellectual style, pacing, and more.
Updated incrementally from every interaction via exponential moving averages,
with periodic deep LLM analysis for pattern refinement.
"""

import json
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.Social")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EMA_ALPHA = 0.15  # Exponential moving average smoothing factor
_DEEP_ANALYSIS_INTERVAL = 20  # Trigger LLM analysis every N interactions
_HUMOR_MARKERS = re.compile(
    r"(lol|lmao|haha|heh|rofl|😂|🤣|😆|💀|dead|dying|hilarious|funny|"
    r"cracking up|that killed me|i can't|im dead)",
    re.IGNORECASE,
)
_NEGATIVE_HUMOR_MARKERS = re.compile(
    r"(not funny|that's not|don't joke|stop|seriously|cringe|yikes|ugh|"
    r"tone deaf|too far|inappropriate)",
    re.IGNORECASE,
)
_VULNERABILITY_MARKERS = re.compile(
    r"(i feel|i'm scared|i'm afraid|i struggle|it hurts|honestly|"
    r"to be honest|vulnerable|i miss|i lost|grief|lonely|anxious|"
    r"depressed|overwhelmed|ashamed|embarrassed|insecure|"
    r"never told anyone|hard to admit|between us|this is personal)",
    re.IGNORECASE,
)
_EMOJI_PATTERN = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    r"\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF"
    r"\U00002702-\U000027B0\U0000FE00-\U0000FE0F\U0001F1E0-\U0001F1FF]"
)
_EXCLAMATION_PATTERN = re.compile(r"!")

# Vocabulary complexity heuristic: longer average word length -> higher complexity
_COMPLEX_WORD_THRESHOLD = 8  # chars

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class ConversationalProfile:
    """Comprehensive per-user communication model."""

    user_id: str

    # --- Communication Style ---
    preferred_response_length: str = "moderate"  # brief | moderate | detailed
    directness_preference: float = 0.5  # 0=indirect/gentle, 1=blunt/direct
    formality_preference: float = 0.3  # 0=very casual, 1=formal
    reasoning_style: str = "pragmatic"  # logical | emotional | pragmatic | aesthetic | intuitive
    confrontation_style: str = "direct"  # avoidant | direct | playful | socratic

    # --- Humor Profile ---
    humor_appreciation: float = 0.5  # 0-1
    humor_types_enjoyed: Dict[str, float] = field(default_factory=lambda: {
        "sarcasm": 0.5,
        "absurdist": 0.5,
        "dry_wit": 0.5,
        "self_deprecating": 0.5,
        "callback": 0.5,
        "puns": 0.5,
        "dark": 0.5,
        "wordplay": 0.5,
        "hyperbole": 0.5,
        "observational": 0.5,
    })
    sarcasm_tolerance: float = 0.5
    irony_sophistication: float = 0.5
    banter_appetite: float = 0.5
    humor_attempts_landed: int = 0
    humor_attempts_missed: int = 0
    favorite_humor_topics: List[str] = field(default_factory=list)

    # --- Emotional / Vulnerability ---
    vulnerability_pace: float = 0.3  # 0-1, how quickly they open up
    disclosure_depth_comfort: float = 0.3  # 0-1
    emotional_support_style: str = "validation"  # validation | solutions | distraction | presence
    needs_when_upset: str = "empathy"  # space | empathy | humor | directness | distraction

    # --- Intellectual ---
    intellectual_curiosity: float = 0.5
    debate_enjoyment: float = 0.5
    topics_of_passion: List[str] = field(default_factory=list)
    topics_that_bore: List[str] = field(default_factory=list)
    preferred_depth: str = "moderate"  # surface | moderate | deep | philosophical

    # --- Pacing ---
    typical_message_length: float = 80.0  # avg chars
    typical_response_gap_s: float = 30.0  # avg seconds between messages
    prefers_rapid_exchange: bool = False
    monologue_tolerance: float = 0.5

    # --- Linguistic Signature ---
    vocabulary_complexity: float = 0.5
    favorite_phrases: List[str] = field(default_factory=list)
    emoji_usage: float = 0.3
    exclamation_frequency: float = 0.3

    # --- Entertainment Value ---
    surprise_appreciation: float = 0.5
    novelty_seeking: float = 0.5
    stimulation_threshold: float = 0.6

    # --- Meta ---
    interactions_analyzed: int = 0
    last_updated: float = field(default_factory=time.time)
    confidence: float = 0.0  # 0-1, grows with interactions

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationalProfile":
        """Reconstruct from serialized dict, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------


class ConversationalProfiler:
    """Builds and incrementally updates per-user conversational profiles."""

    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            try:
                from core.config import config
                storage_path = config.paths.data_dir / "conversational_profiles.json"
            except Exception:
                storage_path = Path.home() / ".aura" / "data" / "conversational_profiles.json"
        self._storage_path = Path(storage_path)
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._profiles: Dict[str, ConversationalProfile] = {}
        self._pacing_history: Dict[str, List[float]] = defaultdict(list)  # user_id -> [timestamps]
        self._phrase_counter: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._load()
        logger.info("ConversationalProfiler initialized (%d profiles loaded)", len(self._profiles))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._storage_path.exists():
            return
        try:
            with open(self._storage_path, "r") as f:
                raw = json.load(f)
            for uid, data in raw.get("profiles", {}).items():
                try:
                    self._profiles[uid] = ConversationalProfile.from_dict(data)
                except Exception as exc:
                    logger.warning("Skipping corrupt profile for '%s': %s", uid, exc)
            # Restore phrase counters if present
            for uid, phrases in raw.get("phrase_counters", {}).items():
                self._phrase_counter[uid] = defaultdict(int, phrases)
        except Exception as exc:
            logger.error("Failed to load conversational profiles: %s", exc)

    def save(self) -> None:
        """Atomic write to disk."""
        try:
            payload = {
                "profiles": {uid: p.to_dict() for uid, p in self._profiles.items()},
                "phrase_counters": {uid: dict(pc) for uid, pc in self._phrase_counter.items()},
            }
            tmp = str(self._storage_path) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, self._storage_path)
            logger.debug("Conversational profiles saved (%d users)", len(self._profiles))
        except Exception as exc:
            logger.error("Failed to save conversational profiles: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_profile(self, user_id: str) -> ConversationalProfile:
        """Get an existing profile or create a fresh one."""
        if user_id not in self._profiles:
            self._profiles[user_id] = ConversationalProfile(user_id=user_id)
            logger.debug("Created new conversational profile for '%s'", user_id)
        return self._profiles[user_id]

    async def update_from_interaction(
        self,
        user_id: str,
        user_message: str,
        aura_response: str,
        reaction_signals: Optional[Dict[str, Any]] = None,
    ) -> ConversationalProfile:
        """Main update entry point. Call after every interaction round-trip.

        Args:
            user_id: Unique user identifier.
            user_message: The user's latest message text.
            aura_response: Aura's response that preceded this user message
                           (i.e., the message the user is *reacting to*).
            reaction_signals: Optional dict with explicit signals:
                - "laughed" (bool), "engaged" (bool), "went_quiet" (bool),
                - "changed_topic" (bool), "expressed_frustration" (bool),
                - "asked_for_more" (bool), "timestamp" (float)
        """
        if reaction_signals is None:
            reaction_signals = {}

        profile = self.get_profile(user_id)
        now = reaction_signals.get("timestamp", time.time())

        # --- Run all heuristic analyzers ---
        self._analyze_message_length(user_message, profile)
        self._analyze_linguistic_signature(user_message, profile)
        self._analyze_emoji_and_exclamation(user_message, profile)
        self._analyze_vocabulary_complexity(user_message, profile)
        self._detect_pacing(user_id, user_message, now)
        self._update_pacing_profile(user_id, profile)

        # Humor detection
        humor_landed = self._detect_humor_reaction(user_message, aura_response)
        if humor_landed is True:
            profile.humor_attempts_landed += 1
            profile.humor_appreciation = _ema(profile.humor_appreciation, 1.0, _EMA_ALPHA)
        elif humor_landed is False:
            profile.humor_attempts_missed += 1
            profile.humor_appreciation = _ema(profile.humor_appreciation, 0.0, _EMA_ALPHA)

        # Vulnerability detection
        vuln_level = self._detect_vulnerability_level(user_message)
        if vuln_level > 0.3:
            profile.vulnerability_pace = _ema(profile.vulnerability_pace, vuln_level, _EMA_ALPHA)
            profile.disclosure_depth_comfort = _ema(
                profile.disclosure_depth_comfort, vuln_level, _EMA_ALPHA * 0.5
            )

        # Reaction signal processing
        self._process_reaction_signals(reaction_signals, profile)

        # Directness heuristic: short messages with declarative tone -> more direct
        if len(user_message.split()) < 8 and not user_message.endswith("?"):
            profile.directness_preference = _ema(profile.directness_preference, 0.8, _EMA_ALPHA * 0.5)
        elif user_message.endswith("?"):
            profile.directness_preference = _ema(profile.directness_preference, 0.4, _EMA_ALPHA * 0.3)

        # Intellectual curiosity: questions, "why", "how does", "explain", "curious"
        curiosity_markers = len(re.findall(
            r"\b(why|how does|explain|curious|fascinating|interesting|wonder|what if)\b",
            user_message, re.IGNORECASE,
        ))
        if curiosity_markers > 0:
            profile.intellectual_curiosity = _ema(
                profile.intellectual_curiosity, min(1.0, 0.5 + curiosity_markers * 0.15), _EMA_ALPHA
            )

        # Update meta
        profile.interactions_analyzed += 1
        profile.last_updated = now
        profile.confidence = min(1.0, profile.interactions_analyzed / 100.0)

        # Periodic deep LLM analysis
        if profile.interactions_analyzed % _DEEP_ANALYSIS_INTERVAL == 0:
            await self._deep_llm_analysis(user_id, profile)

        # Persist every 5 interactions to avoid excessive I/O
        if profile.interactions_analyzed % 5 == 0:
            self.save()

        return profile

    def get_context_injection(self, user_id: str) -> str:
        """Produce a formatted markdown block for LLM system prompt injection."""
        profile = self.get_profile(user_id)

        if profile.interactions_analyzed < 3:
            return ""  # Not enough data yet

        # Style summary
        length_desc = {
            "brief": "values brevity",
            "moderate": "balanced message length",
            "detailed": "appreciates detailed responses",
        }.get(profile.preferred_response_length, "balanced")

        directness_desc = (
            "very direct" if profile.directness_preference > 0.7
            else "indirect/diplomatic" if profile.directness_preference < 0.3
            else "moderately direct"
        )

        formality_desc = (
            "formal" if profile.formality_preference > 0.7
            else "casual" if profile.formality_preference < 0.3
            else "semi-casual"
        )

        # Humor summary
        humor_total = profile.humor_attempts_landed + profile.humor_attempts_missed
        humor_rate = (
            f"{round(profile.humor_attempts_landed / humor_total * 100)}% landing rate"
            if humor_total > 0
            else "not enough data"
        )

        top_humor = sorted(
            profile.humor_types_enjoyed.items(), key=lambda x: x[1], reverse=True
        )[:3]
        humor_types_desc = ", ".join(
            t.replace("_", " ") for t, s in top_humor if s > 0.5
        )
        if not humor_types_desc:
            humor_types_desc = "still learning preferences"

        sarcasm_desc = (
            "High sarcasm tolerance" if profile.sarcasm_tolerance > 0.7
            else "Low sarcasm tolerance" if profile.sarcasm_tolerance < 0.3
            else "Moderate sarcasm tolerance"
        )

        banter_desc = (
            "Enjoys banter" if profile.banter_appetite > 0.6
            else "Not big on banter" if profile.banter_appetite < 0.3
            else "Open to some banter"
        )

        # Emotional support
        upset_map = {
            "space": "Prefers space when upset",
            "empathy": "Wants empathy when upset",
            "humor": "Uses humor to cope",
            "directness": "Prefers directness over coddling",
            "distraction": "Prefers distraction when upset",
        }
        upset_desc = upset_map.get(profile.needs_when_upset, "Responds to empathy")

        # Intellectual
        depth_map = {
            "surface": "Prefers surface-level discussion",
            "moderate": "Comfortable at moderate depth",
            "deep": "Deep thinker",
            "philosophical": "Loves philosophical depth",
        }
        depth_desc = depth_map.get(profile.preferred_depth, "Moderate depth")

        passions = ", ".join(profile.topics_of_passion[:5]) if profile.topics_of_passion else "still discovering"
        bores = ", ".join(profile.topics_that_bore[:3]) if profile.topics_that_bore else "unknown"

        # Pacing
        pacing_desc = (
            "Tends toward short messages"
            if profile.typical_message_length < 60
            else "Writes longer messages"
            if profile.typical_message_length > 200
            else "Medium-length messages"
        )
        rapid_desc = "Appreciates rapid exchange" if profile.prefers_rapid_exchange else "Comfortable with measured pace"

        # Engagement
        novelty_desc = (
            "Loves novelty and surprise"
            if profile.novelty_seeking > 0.7
            else "Prefers familiar territory"
            if profile.novelty_seeking < 0.3
            else "Open to new ideas"
        )

        lines = [
            f"## WHO {user_id.upper()} IS (Communication DNA)",
            f"- **Style**: {directness_desc.capitalize()}, {formality_desc}, {length_desc}",
            f"- **Reasoning**: {profile.reasoning_style.capitalize()} thinker, {profile.confrontation_style} confrontation style",
            f"- **Humor**: Loves {humor_types_desc} ({humor_rate}). {sarcasm_desc}. {banter_desc}.",
            f"- **When upset**: {upset_desc}. Don't over-empathize." if profile.needs_when_upset == "directness" else f"- **When upset**: {upset_desc}.",
            f"- **Intellectually**: {depth_desc}. Passionate about {passions}."
            + (f" Gets bored by {bores}." if profile.topics_that_bore else ""),
            f"- **Pacing**: {pacing_desc}. {rapid_desc}.",
            f"- **What engages**: {novelty_desc}. Curiosity level: {_label_01(profile.intellectual_curiosity)}.",
        ]

        if profile.favorite_phrases:
            lines.append(
                f"- **Their phrases**: {', '.join(repr(p) for p in profile.favorite_phrases[:5])}"
            )

        lines.append(f"- **Profile confidence**: {_label_01(profile.confidence)}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _detect_humor_reaction(
        self, user_message: str, prev_aura_message: str
    ) -> Optional[bool]:
        """Did our humor land? Returns True/False/None (no humor attempted)."""
        if not prev_aura_message:
            return None

        # Heuristic: Aura's message likely contained humor if it had humor markers
        # or was playful. We check if user's *reaction* signals appreciation or rejection.
        aura_lower = prev_aura_message.lower()
        aura_humor_signals = any(
            marker in aura_lower
            for marker in ("haha", "😄", "😏", "🤣", "joke", ";)", "😉", "lol")
        )
        # Also detect humor through sentence structure (short quips, rhetorical questions)
        aura_has_quip = bool(re.search(r"[.!]\s*\.\.\.", prev_aura_message))

        if not aura_humor_signals and not aura_has_quip:
            return None  # No humor was attempted

        user_lower = user_message.lower()
        if _HUMOR_MARKERS.search(user_lower):
            return True
        if _NEGATIVE_HUMOR_MARKERS.search(user_lower):
            return False

        return None  # Ambiguous

    def _detect_vulnerability_level(self, message: str) -> float:
        """Score 0-1 how vulnerable/emotionally open this message is."""
        matches = len(_VULNERABILITY_MARKERS.findall(message))
        if matches == 0:
            return 0.0
        # Scale: 1 match ~0.4, 2 ~0.6, 3+ ~0.8+
        return min(1.0, 0.2 + matches * 0.2)

    def _analyze_linguistic_signature(
        self, message: str, profile: ConversationalProfile
    ) -> None:
        """Track phrase frequency and update favorite_phrases."""
        user_id = profile.user_id
        # Extract 2-4 word phrases (n-grams)
        words = message.lower().split()
        for n in (2, 3, 4):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i : i + n])
                # Skip very common phrases
                if phrase in _COMMON_PHRASES:
                    continue
                self._phrase_counter[user_id][phrase] += 1

        # Rebuild favorite phrases from top counts (minimum 3 occurrences)
        counter = self._phrase_counter[user_id]
        top = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        profile.favorite_phrases = [
            phrase for phrase, count in top[:10] if count >= 3
        ]

    def _analyze_message_length(
        self, message: str, profile: ConversationalProfile
    ) -> None:
        """Update typical message length and infer preferred response length."""
        msg_len = float(len(message))
        profile.typical_message_length = _ema(
            profile.typical_message_length, msg_len, _EMA_ALPHA
        )
        # Infer preferred response length from user's own message length
        if profile.typical_message_length < 50:
            profile.preferred_response_length = "brief"
        elif profile.typical_message_length > 250:
            profile.preferred_response_length = "detailed"
        else:
            profile.preferred_response_length = "moderate"

    def _analyze_emoji_and_exclamation(
        self, message: str, profile: ConversationalProfile
    ) -> None:
        """Track emoji and exclamation usage rates."""
        word_count = max(1, len(message.split()))
        emoji_count = len(_EMOJI_PATTERN.findall(message))
        excl_count = len(_EXCLAMATION_PATTERN.findall(message))

        emoji_rate = min(1.0, emoji_count / word_count)
        excl_rate = min(1.0, excl_count / max(1, message.count(".")))

        profile.emoji_usage = _ema(profile.emoji_usage, emoji_rate, _EMA_ALPHA)
        profile.exclamation_frequency = _ema(
            profile.exclamation_frequency, excl_rate, _EMA_ALPHA
        )

        # Emoji/exclamation usage inversely correlates with formality
        if emoji_rate > 0.1 or excl_rate > 0.5:
            profile.formality_preference = _ema(
                profile.formality_preference, 0.1, _EMA_ALPHA * 0.3
            )

    def _analyze_vocabulary_complexity(
        self, message: str, profile: ConversationalProfile
    ) -> None:
        """Heuristic vocabulary complexity from average word length and long-word ratio."""
        words = re.findall(r"[a-zA-Z]+", message)
        if not words:
            return
        avg_len = sum(len(w) for w in words) / len(words)
        long_ratio = sum(1 for w in words if len(w) >= _COMPLEX_WORD_THRESHOLD) / len(words)
        # Normalize: avg_len of 4 -> 0.2, avg_len of 8 -> 0.8
        complexity = min(1.0, max(0.0, (avg_len - 3.0) / 6.0) * 0.6 + long_ratio * 0.4)
        profile.vocabulary_complexity = _ema(
            profile.vocabulary_complexity, complexity, _EMA_ALPHA
        )

    def _detect_pacing(
        self, user_id: str, message: str, timestamp: float
    ) -> None:
        """Record message timestamp for pacing analysis."""
        self._pacing_history[user_id].append(timestamp)
        # Keep only last 50 timestamps
        if len(self._pacing_history[user_id]) > 50:
            self._pacing_history[user_id] = self._pacing_history[user_id][-50:]

    def _update_pacing_profile(
        self, user_id: str, profile: ConversationalProfile
    ) -> None:
        """Compute average gap and rapid-exchange preference from pacing history."""
        timestamps = self._pacing_history.get(user_id, [])
        if len(timestamps) < 2:
            return
        gaps = [
            timestamps[i] - timestamps[i - 1]
            for i in range(1, len(timestamps))
            if 0 < (timestamps[i] - timestamps[i - 1]) < 3600  # Ignore gaps > 1hr (session breaks)
        ]
        if not gaps:
            return
        avg_gap = sum(gaps) / len(gaps)
        profile.typical_response_gap_s = _ema(
            profile.typical_response_gap_s, avg_gap, _EMA_ALPHA
        )
        profile.prefers_rapid_exchange = avg_gap < 15.0

    def _process_reaction_signals(
        self, signals: Dict[str, Any], profile: ConversationalProfile
    ) -> None:
        """Interpret explicit reaction signals from the interaction layer."""
        if not signals:
            return

        if signals.get("laughed"):
            profile.humor_appreciation = _ema(profile.humor_appreciation, 1.0, _EMA_ALPHA)
            profile.banter_appetite = _ema(profile.banter_appetite, 0.7, _EMA_ALPHA * 0.5)

        if signals.get("engaged"):
            profile.intellectual_curiosity = _ema(
                profile.intellectual_curiosity, 0.8, _EMA_ALPHA * 0.5
            )
            profile.novelty_seeking = _ema(profile.novelty_seeking, 0.7, _EMA_ALPHA * 0.5)

        if signals.get("went_quiet"):
            profile.stimulation_threshold = _ema(
                profile.stimulation_threshold, 0.3, _EMA_ALPHA * 0.5
            )
            profile.monologue_tolerance = _ema(
                profile.monologue_tolerance, 0.2, _EMA_ALPHA * 0.3
            )

        if signals.get("changed_topic"):
            # Possible boredom or discomfort
            profile.monologue_tolerance = _ema(
                profile.monologue_tolerance, 0.3, _EMA_ALPHA * 0.3
            )

        if signals.get("expressed_frustration"):
            profile.needs_when_upset = "directness"
            profile.directness_preference = _ema(
                profile.directness_preference, 0.8, _EMA_ALPHA
            )

        if signals.get("asked_for_more"):
            profile.preferred_depth = "deep"
            profile.intellectual_curiosity = _ema(
                profile.intellectual_curiosity, 0.9, _EMA_ALPHA
            )

        # Humor type signals (e.g., {"humor_type": "sarcasm", "landed": True})
        humor_type = signals.get("humor_type")
        if humor_type and humor_type in profile.humor_types_enjoyed:
            landed = signals.get("landed", True)
            target = 1.0 if landed else 0.0
            profile.humor_types_enjoyed[humor_type] = _ema(
                profile.humor_types_enjoyed[humor_type], target, _EMA_ALPHA
            )

        # Topic signals
        passion_topic = signals.get("topic_passion")
        if passion_topic and passion_topic not in profile.topics_of_passion:
            profile.topics_of_passion.append(passion_topic)
            profile.topics_of_passion = profile.topics_of_passion[-20:]  # Cap

        bore_topic = signals.get("topic_bore")
        if bore_topic and bore_topic not in profile.topics_that_bore:
            profile.topics_that_bore.append(bore_topic)
            profile.topics_that_bore = profile.topics_that_bore[-10:]

    # ------------------------------------------------------------------
    # Deep LLM analysis (periodic)
    # ------------------------------------------------------------------

    async def _deep_llm_analysis(
        self, user_id: str, profile: ConversationalProfile
    ) -> None:
        """Use LLM to refine profile fields that are hard to detect heuristically."""
        try:
            from core.container import ServiceContainer

            brain = ServiceContainer.get(
                "cognitive_integration",
                default=ServiceContainer.get("cognitive_engine", default=None),
            )
            if not brain:
                logger.debug("Deep profile analysis skipped: no cognitive engine available")
                return

            current = profile.to_dict()
            prompt = (
                "You are analyzing a user's conversational profile to refine it.\n"
                "Current profile snapshot (JSON):\n"
                f"```json\n{json.dumps(current, indent=2)}\n```\n\n"
                "Based on the accumulated statistics, suggest JSON updates for these fields ONLY:\n"
                "- reasoning_style (logical/emotional/pragmatic/aesthetic/intuitive)\n"
                "- confrontation_style (avoidant/direct/playful/socratic)\n"
                "- emotional_support_style (validation/solutions/distraction/presence)\n"
                "- needs_when_upset (space/empathy/humor/directness/distraction)\n"
                "- preferred_depth (surface/moderate/deep/philosophical)\n"
                "\nReturn ONLY a JSON object with the fields you want to update. "
                "Omit fields that should stay the same."
            )

            thought = await brain.think(
                objective=prompt,
                context={"user_id": user_id, "profile": current},
                mode="FAST",
            )

            from core.utils.json_utils import extract_json

            updates = extract_json(thought.content)
            if updates and isinstance(updates, dict):
                allowed = {
                    "reasoning_style",
                    "confrontation_style",
                    "emotional_support_style",
                    "needs_when_upset",
                    "preferred_depth",
                }
                for key, value in updates.items():
                    if key in allowed and isinstance(value, str):
                        setattr(profile, key, value)
                logger.info(
                    "Deep profile analysis applied %d updates for '%s'",
                    len(updates),
                    user_id,
                )
        except Exception as exc:
            logger.debug("Deep profile analysis failed for '%s': %s", user_id, exc)

    # ------------------------------------------------------------------
    # ServiceContainer integration
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """Called by ServiceContainer on initialization."""
        pass

    async def on_start_async(self) -> None:
        """Async lifecycle hook."""
        pass


# ---------------------------------------------------------------------------
# Common phrases to ignore in n-gram extraction
# ---------------------------------------------------------------------------

_COMMON_PHRASES = frozenset({
    "i think", "it is", "i am", "i have", "do you", "can you", "what is",
    "this is", "that is", "there is", "i don't", "i dont", "how to",
    "want to", "need to", "going to", "have to", "it was", "i was",
    "you can", "you are", "we can", "the same", "as well", "a lot",
    "in the", "on the", "at the", "for the", "to the", "of the",
    "and the", "is a", "is the", "i will", "i would", "i could",
    "thank you", "thanks for", "i know", "you know", "let me",
})


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _ema(current: float, observation: float, alpha: float) -> float:
    """Exponential moving average: smooth update toward observation."""
    return current * (1.0 - alpha) + observation * alpha


def _label_01(value: float) -> str:
    """Human-readable label for a 0-1 float."""
    if value >= 0.8:
        return "very high"
    if value >= 0.6:
        return "high"
    if value >= 0.4:
        return "moderate"
    if value >= 0.2:
        return "low"
    return "very low"


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_profiler_instance: Optional[ConversationalProfiler] = None


def get_conversational_profiler() -> ConversationalProfiler:
    """Return the singleton ConversationalProfiler, creating it if needed."""
    global _profiler_instance
    if _profiler_instance is None:
        _profiler_instance = ConversationalProfiler()
        # Register with ServiceContainer if not already there
        try:
            from core.container import ServiceContainer

            if not ServiceContainer.has("conversational_profiler"):
                ServiceContainer.register_instance(
                    "conversational_profiler", _profiler_instance
                )
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)
    return _profiler_instance
