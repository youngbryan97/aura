"""core/conversational/conversation_intelligence.py
──────────────────────────────────────────────────────
Conversation Intelligence Engine.

Meta-conversational awareness: the skills humans deploy unconsciously
to navigate a conversation. Arc tracking, silence intelligence, pacing
calibration, topic trajectory prediction, and code-switching.

This is the layer ABOVE dynamics and discourse tracking. It consumes
their output and produces a unified ConversationalDirective that tells
the response generator HOW to respond — not what to say, but how long,
how fast, in what register, and whether to speak at all.

Based on:
  - Conversation Analysis (Sacks, Schegloff, Jefferson 1974) — sequential structure
  - Accommodation Theory (Giles 1973) — register convergence
  - Clark & Schaefer (1989) — grounding and mutual understanding
  - Tannen (2005) — conversational style (high/low involvement)

Register in ServiceContainer as "conversation_intelligence".
"""
from __future__ import annotations

import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.ConversationIntelligence")

# ── Farewell / pause signals ────────────────────────────────────────────────

_FAREWELL_PATTERNS = re.compile(
    r'\b(?:bye|goodbye|good night|goodnight|night|gotta go|gtg|ttyl|later|'
    r'see you|see ya|talk later|peace|take care|signing off|logging off|'
    r'heading out|i\'m out|cya)\b',
    re.IGNORECASE,
)

_PAUSE_PATTERNS = re.compile(
    r'\b(?:brb|one sec|hold on|give me a sec|gimme a sec|be right back|'
    r'one moment|hang on|wait)\b',
    re.IGNORECASE,
)

_EMOTIONAL_HEAVY_PATTERNS = re.compile(
    r'\b(?:i\'m not okay|not doing great|struggling|i feel like crying|'
    r'i don\'t know what to do|everything is falling apart|i\'m scared|'
    r'i feel so alone|i can\'t do this|it\'s too much|i\'m breaking|'
    r'kind of hard right now|honestly.*scared|i need help|i\'m lost|'
    r'i don\'t want to be here|i feel empty|it hurts)\b',
    re.IGNORECASE,
)

_EMOJI_PATTERN = re.compile(
    r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
    r'\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF'
    r'\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]'
)

_SLANG_MARKERS = frozenset({
    "lol", "lmao", "bruh", "bro", "ngl", "fr", "imo", "tbh", "idk",
    "nah", "yep", "dude", "lowkey", "highkey", "deadass", "bet", "fam",
    "ong", "wyd", "smh", "istg", "periodt", "slay", "bussin", "no cap",
})


# ── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass
class ConversationArc:
    """Tracks the macro shape of a conversation."""
    phase: str = "opening"  # opening, building, peak, winding_down, closing
    turns_in_phase: int = 0
    total_turns: int = 0
    energy_trajectory: List[float] = field(default_factory=list)  # last 10 readings
    peak_moment: Optional[str] = None  # what the conversation peaked on
    emotional_high_point: float = 0.0
    topics_covered: int = 0
    depth_reached: str = "surface"  # surface, moderate, deep, intimate


@dataclass
class SilenceIntelligence:
    """When NOT to speak — and why."""
    should_be_quiet: bool = False
    reason: str = "none"
    # "user_processing", "comfortable_pause", "user_typing", "emotional_space",
    # "topic_exhaustion", "natural_end"
    suggested_wait_s: float = 0.0
    silence_type: str = "comfortable"  # comfortable, loaded, thinking, exhausted


@dataclass
class PacingDirective:
    """Calibrates response length and rhythm."""
    target_length: str = "moderate"  # minimal, brief, moderate, detailed, expansive
    target_chars: int = 200
    energy_match: str = "match_user"  # match_user, lift_energy, calm_energy, maintain
    cadence: str = "conversational"  # rapid_fire, conversational, considered, reflective
    sentence_count_target: int = 3


@dataclass
class TopicTrajectory:
    """Predicts where the conversation should go next."""
    predicted_next_topic: Optional[str] = None
    user_likely_wants: str = "go_deeper"
    # go_deeper, change_topic, wrap_up, play, vent, explore, be_quiet
    suggested_transition: Optional[str] = None  # natural bridge phrase
    topics_to_avoid: List[str] = field(default_factory=list)  # exhausted or sensitive
    topics_to_surface: List[str] = field(default_factory=list)  # from interests + open threads


@dataclass
class CodeSwitchContext:
    """Register and vocabulary adaptation."""
    recommended_register: str = "casual"  # formal, casual, intimate, playful, professional
    vocabulary_level: str = "moderate"  # simple, moderate, sophisticated, technical
    should_mirror_emoji: bool = False
    should_use_slang: bool = False
    formality_shift_reason: Optional[str] = None


@dataclass
class ConversationalDirective:
    """The unified output — everything the response generator needs."""
    arc: ConversationArc = field(default_factory=ConversationArc)
    silence: SilenceIntelligence = field(default_factory=SilenceIntelligence)
    pacing: PacingDirective = field(default_factory=PacingDirective)
    trajectory: TopicTrajectory = field(default_factory=TopicTrajectory)
    code_switch: CodeSwitchContext = field(default_factory=CodeSwitchContext)
    timestamp: float = field(default_factory=time.time)


# ── Transition bridge phrases ───────────────────────────────────────────────

_TRANSITION_BRIDGES = {
    "deeper":      ["Tell me more about that.", "What makes you say that?", "Go on."],
    "lighter":     ["On a lighter note...", "Anyway —", "Speaking of which..."],
    "callback":    ["Actually, going back to what you said about {topic}...",
                    "That reminds me — you mentioned {topic} earlier."],
    "wind_down":   ["This was good.", "I'm glad we talked about this."],
    "topic_shift": ["Oh — that reminds me.", "Completely different thing —",
                    "Random thought:"],
}


# ── Engine ──────────────────────────────────────────────────────────────────

class ConversationIntelligenceEngine:
    """
    Computes meta-conversational intelligence from dynamics + discourse state.
    Produces a ConversationalDirective on each turn that feeds the system prompt.
    """

    MAX_ENERGY_HISTORY = 10
    MAX_TIMESTAMPS = 30
    MAX_LENGTHS = 20

    def __init__(self):
        self._arc = ConversationArc()
        self._directive = ConversationalDirective()
        self._message_timestamps: deque = deque(maxlen=self.MAX_TIMESTAMPS)
        self._user_message_lengths: deque = deque(maxlen=self.MAX_LENGTHS)
        self._aura_message_lengths: deque = deque(maxlen=self.MAX_LENGTHS)
        self._seen_topics: set = set()
        self._topic_dwell_counts: Dict[str, int] = {}  # topic -> turns spent on it
        self._last_user_message: str = ""
        self._last_energy: float = 0.5
        self._phase_entered_at_turn: int = 0

    # ── Main update ──────────────────────────────────────────────────────

    async def update(
        self,
        user_message: str,
        aura_response: str,
        dynamics_state: Any,
        discourse_state: Dict,
    ) -> ConversationalDirective:
        """
        The main method. Runs after each exchange. Computes all sub-signals
        and returns a unified ConversationalDirective.

        Parameters
        ----------
        user_message : str
            The user's latest message.
        aura_response : str
            Aura's latest response (may be empty on first call).
        dynamics_state : ConversationalDynamicsState
            Output from the conversational dynamics engine.
        discourse_state : dict
            Fields from DiscourseTracker (topic, depth, energy, trend, branches).
        """
        now = time.time()
        self._message_timestamps.append(now)
        self._user_message_lengths.append(len(user_message))
        if aura_response:
            self._aura_message_lengths.append(len(aura_response))
        self._last_user_message = user_message

        # Extract discourse fields with safe defaults
        conversation_energy = discourse_state.get("conversation_energy", 0.5)
        discourse_depth = discourse_state.get("discourse_depth", 1)
        discourse_topic = discourse_state.get("discourse_topic", "general")
        user_trend = discourse_state.get("user_emotional_trend", "neutral")
        branches = discourse_state.get("discourse_branches", [])

        # Track topic coverage
        if discourse_topic and discourse_topic != "general":
            self._seen_topics.add(discourse_topic)
            self._topic_dwell_counts[discourse_topic] = (
                self._topic_dwell_counts.get(discourse_topic, 0) + 1
            )

        # Compute sub-signals
        arc = self._compute_arc(conversation_energy, discourse_depth, dynamics_state)
        silence = self._compute_silence(user_message, dynamics_state, conversation_energy, user_trend)
        pacing = self._compute_pacing(user_message, dynamics_state, conversation_energy)
        trajectory = self._compute_trajectory(
            dynamics_state, discourse_topic, conversation_energy, user_trend, branches
        )
        code_switch = self._compute_code_switch(user_message, dynamics_state, arc)

        self._directive = ConversationalDirective(
            arc=arc,
            silence=silence,
            pacing=pacing,
            trajectory=trajectory,
            code_switch=code_switch,
            timestamp=now,
        )
        self._last_energy = conversation_energy
        return self._directive

    # ── Arc tracking ─────────────────────────────────────────────────────

    def _compute_arc(
        self,
        energy: float,
        depth: int,
        dynamics_state: Any,
    ) -> ConversationArc:
        """Detect conversation phase from energy trajectory + turn count + topic density."""
        prev = self._arc
        total_turns = prev.total_turns + 1

        # Update energy trajectory (rolling window of 10)
        trajectory = list(prev.energy_trajectory)
        trajectory.append(energy)
        if len(trajectory) > self.MAX_ENERGY_HISTORY:
            trajectory = trajectory[-self.MAX_ENERGY_HISTORY:]

        # Track peak
        peak_moment = prev.peak_moment
        emotional_high = prev.emotional_high_point
        partner_intensity = getattr(dynamics_state, "partner_intensity", 0.5)
        if partner_intensity > emotional_high:
            emotional_high = partner_intensity
            topic = getattr(dynamics_state, "current_topic", "general")
            peak_moment = topic

        # Determine depth_reached from discourse depth and frame
        partner_frame = getattr(dynamics_state, "partner_frame", "neutral")
        if partner_frame in ("vulnerable",) or depth > 12:
            depth_label = "intimate"
        elif depth > 8 or partner_frame in ("serious",):
            depth_label = "deep"
        elif depth > 4:
            depth_label = "moderate"
        else:
            depth_label = "surface"

        # Phase detection
        phase = self._detect_phase(
            prev.phase, total_turns, trajectory, energy, dynamics_state
        )
        if phase != prev.phase:
            turns_in_phase = 1
            self._phase_entered_at_turn = total_turns
        else:
            turns_in_phase = prev.turns_in_phase + 1

        arc = ConversationArc(
            phase=phase,
            turns_in_phase=turns_in_phase,
            total_turns=total_turns,
            energy_trajectory=trajectory,
            peak_moment=peak_moment,
            emotional_high_point=emotional_high,
            topics_covered=len(self._seen_topics),
            depth_reached=depth_label,
        )
        self._arc = arc
        return arc

    def _detect_phase(
        self,
        current_phase: str,
        total_turns: int,
        trajectory: List[float],
        energy: float,
        dynamics_state: Any,
    ) -> str:
        """
        Phase transitions:
        - opening: first 3 turns
        - building: energy rising, topics deepening
        - peak: highest energy or deepest vulnerability
        - winding_down: energy declining, topics broadening
        - closing: low energy, short messages, farewell signals
        """
        msg = self._last_user_message
        msg_len = len(msg)

        # Farewell signals immediately trigger closing
        if _FAREWELL_PATTERNS.search(msg):
            return "closing"

        # Opening: first 3 turns
        if total_turns <= 3:
            return "opening"

        # Compute energy trend from trajectory
        if len(trajectory) >= 3:
            recent = trajectory[-3:]
            older = trajectory[-min(len(trajectory), 6):-3] if len(trajectory) > 3 else [0.5]
            avg_recent = sum(recent) / len(recent)
            avg_older = sum(older) / max(len(older), 1)
            energy_delta = avg_recent - avg_older
        else:
            energy_delta = 0.0
            avg_recent = energy

        partner_frame = getattr(dynamics_state, "partner_frame", "neutral")
        partner_intensity = getattr(dynamics_state, "partner_intensity", 0.5)

        # Closing: sustained low energy + short messages
        if avg_recent < 0.25 and msg_len < 40 and current_phase in ("winding_down", "closing"):
            return "closing"

        # Peak: high vulnerability or peak intensity moment
        if (partner_frame in ("vulnerable", "serious") and partner_intensity > 0.7) or \
           (energy > 0.8 and energy_delta > 0.0):
            if current_phase in ("building", "peak"):
                return "peak"

        # Winding down: energy declining after peak
        if current_phase == "peak" and energy_delta < -0.05:
            return "winding_down"

        if current_phase == "winding_down":
            # Stay winding down unless energy recovers
            if energy_delta > 0.1:
                return "building"
            return "winding_down"

        # Building: energy rising or deepening
        if energy_delta > 0.02 or (energy > 0.4 and current_phase == "opening"):
            return "building"

        # Default: stay in current phase
        if current_phase in ("building", "peak"):
            # Allow natural progression
            if energy_delta < -0.08:
                return "winding_down"
            return current_phase

        return current_phase if current_phase != "opening" else "building"

    # ── Silence intelligence ─────────────────────────────────────────────

    def _compute_silence(
        self,
        user_message: str,
        dynamics_state: Any,
        energy: float,
        user_trend: str,
    ) -> SilenceIntelligence:
        """Determine if Aura should NOT respond or should wait."""

        multi_stream = getattr(dynamics_state, "multi_message_stream", False)
        topic_exhaustion = getattr(dynamics_state, "topic_exhaustion", 0.0)
        partner_frame = getattr(dynamics_state, "partner_frame", "neutral")
        partner_intensity = getattr(dynamics_state, "partner_intensity", 0.5)

        # 1. User is still typing (multi-message stream)
        if multi_stream:
            return SilenceIntelligence(
                should_be_quiet=True,
                reason="user_typing",
                suggested_wait_s=3.0,
                silence_type="thinking",
            )

        # 2. User just said "brb", "one sec", etc.
        if _PAUSE_PATTERNS.search(user_message):
            return SilenceIntelligence(
                should_be_quiet=True,
                reason="comfortable_pause",
                suggested_wait_s=0.0,  # wait indefinitely until they return
                silence_type="comfortable",
            )

        # 3. Deeply emotional message — give space before responding
        if _EMOTIONAL_HEAVY_PATTERNS.search(user_message):
            return SilenceIntelligence(
                should_be_quiet=False,  # don't skip responding, but pause first
                reason="emotional_space",
                suggested_wait_s=4.0,
                silence_type="loaded",
            )

        # 4. High vulnerability — brief pause to not seem instant/robotic
        if partner_frame == "vulnerable" and partner_intensity > 0.6:
            return SilenceIntelligence(
                should_be_quiet=False,
                reason="emotional_space",
                suggested_wait_s=3.0,
                silence_type="loaded",
            )

        # 5. Topic exhaustion — conversation is circling
        if topic_exhaustion > 0.7:
            return SilenceIntelligence(
                should_be_quiet=False,
                reason="topic_exhaustion",
                suggested_wait_s=1.0,
                silence_type="exhausted",
            )

        # 6. Energy near zero + farewell signals = natural end
        if energy < 0.15 and _FAREWELL_PATTERNS.search(user_message):
            return SilenceIntelligence(
                should_be_quiet=False,
                reason="natural_end",
                suggested_wait_s=0.0,
                silence_type="comfortable",
            )

        # 7. Cooling off trend + low energy = user may be processing
        if user_trend == "cooling_off" and energy < 0.3:
            return SilenceIntelligence(
                should_be_quiet=False,
                reason="user_processing",
                suggested_wait_s=2.0,
                silence_type="thinking",
            )

        # Default: no silence needed
        return SilenceIntelligence(
            should_be_quiet=False,
            reason="none",
            suggested_wait_s=0.0,
            silence_type="comfortable",
        )

    # ── Pacing ───────────────────────────────────────────────────────────

    def _compute_pacing(
        self,
        user_message: str,
        dynamics_state: Any,
        energy: float,
    ) -> PacingDirective:
        """Calibrate response length and rhythm to match user patterns."""

        msg_len = len(user_message)
        partner_intensity = getattr(dynamics_state, "partner_intensity", 0.5)
        partner_frame = getattr(dynamics_state, "partner_frame", "neutral")

        # Compute average user message length
        if self._user_message_lengths:
            avg_user_len = sum(self._user_message_lengths) / len(self._user_message_lengths)
        else:
            avg_user_len = msg_len

        # Compute message gap (time between last two messages)
        msg_gap = self._compute_recent_gap()

        # ── Rapid-fire detection ──
        # Messages < 30 chars with gap < 10s = rapid-fire mode
        if msg_len < 30 and msg_gap is not None and msg_gap < 10.0:
            return PacingDirective(
                target_length="minimal",
                target_chars=min(50, msg_len + 20),
                energy_match="match_user",
                cadence="rapid_fire",
                sentence_count_target=1,
            )

        # ── Reflective mode ──
        # Long messages (> 500 chars) = user is being expansive, we can go deeper
        if msg_len > 500:
            return PacingDirective(
                target_length="detailed",
                target_chars=min(800, int(msg_len * 0.8)),
                energy_match="match_user",
                cadence="reflective",
                sentence_count_target=5,
            )

        # ── Length matching ±30% ──
        # Target is proportional to user's message, bounded by their average
        base_target = int(avg_user_len * 1.0)  # match their average
        target_chars = max(40, min(600, base_target))

        # Adjust for emotional weight
        if partner_frame in ("vulnerable", "serious"):
            # Don't overwhelm — keep it measured
            target_chars = min(target_chars, 300)
            cadence = "considered"
            energy_match = "calm_energy"
        elif partner_frame == "playful" or energy > 0.7:
            cadence = "conversational"
            energy_match = "lift_energy" if energy < 0.5 else "match_user"
        elif energy < 0.3:
            cadence = "considered"
            energy_match = "maintain"
            target_chars = min(target_chars, 200)
        else:
            cadence = "conversational"
            energy_match = "match_user"

        # Derive target_length label
        if target_chars < 60:
            target_length = "minimal"
        elif target_chars < 150:
            target_length = "brief"
        elif target_chars < 350:
            target_length = "moderate"
        elif target_chars < 600:
            target_length = "detailed"
        else:
            target_length = "expansive"

        # Sentence count heuristic: ~60 chars per sentence
        sentence_target = max(1, min(8, target_chars // 60))

        return PacingDirective(
            target_length=target_length,
            target_chars=target_chars,
            energy_match=energy_match,
            cadence=cadence,
            sentence_count_target=sentence_target,
        )

    def _compute_recent_gap(self) -> Optional[float]:
        """Compute seconds between the two most recent message timestamps."""
        if len(self._message_timestamps) < 2:
            return None
        return self._message_timestamps[-1] - self._message_timestamps[-2]

    # ── Topic trajectory ─────────────────────────────────────────────────

    def _compute_trajectory(
        self,
        dynamics_state: Any,
        discourse_topic: str,
        energy: float,
        user_trend: str,
        branches: List[str],
    ) -> TopicTrajectory:
        """Predict where conversation should go next."""

        topic_exhaustion = getattr(dynamics_state, "topic_exhaustion", 0.0)
        open_threads = getattr(dynamics_state, "open_threads", [])
        partner_frame = getattr(dynamics_state, "partner_frame", "neutral")
        partner_intensity = getattr(dynamics_state, "partner_intensity", 0.5)
        current_topic = getattr(dynamics_state, "current_topic", discourse_topic)

        # ── Determine user_likely_wants ──
        if _FAREWELL_PATTERNS.search(self._last_user_message):
            user_wants = "wrap_up"
        elif partner_frame == "vulnerable":
            user_wants = "vent"
        elif partner_frame == "playful" and partner_intensity > 0.5:
            user_wants = "play"
        elif topic_exhaustion > 0.7:
            user_wants = "change_topic"
        elif energy < 0.2 and user_trend == "cooling_off":
            user_wants = "be_quiet"
        elif energy > 0.6 and partner_intensity > 0.5:
            user_wants = "go_deeper"
        elif branches:
            user_wants = "explore"
        else:
            user_wants = "go_deeper"

        # ── Topics to avoid (exhausted) ──
        topics_to_avoid = [
            topic for topic, count in self._topic_dwell_counts.items()
            if count > 8  # too many turns on one topic
        ]
        if topic_exhaustion > 0.7 and current_topic:
            if current_topic not in topics_to_avoid:
                topics_to_avoid.append(current_topic)

        # ── Topics to surface ──
        topics_to_surface = []
        # From open threads — most urgent first
        if open_threads:
            urgent = sorted(open_threads, key=lambda t: -getattr(t, "urgency", 0.5))
            for thread in urgent[:3]:
                content = getattr(thread, "content", "")
                if content and content not in topics_to_surface:
                    topics_to_surface.append(content[:60])

        # From discourse branches
        for branch in branches[:2]:
            if branch not in topics_to_avoid and branch not in topics_to_surface:
                topics_to_surface.append(branch)

        # ── Predicted next topic ──
        predicted_next = None
        if user_wants == "change_topic" and topics_to_surface:
            predicted_next = topics_to_surface[0]
        elif user_wants == "explore" and branches:
            predicted_next = branches[0]

        # ── Suggested transition ──
        suggested_transition = None
        if user_wants == "change_topic":
            if topics_to_surface:
                template = _TRANSITION_BRIDGES["callback"][0]
                suggested_transition = template.format(topic=topics_to_surface[0])
            else:
                suggested_transition = _TRANSITION_BRIDGES["topic_shift"][0]
        elif user_wants == "wrap_up":
            suggested_transition = _TRANSITION_BRIDGES["wind_down"][0]
        elif user_wants == "go_deeper":
            suggested_transition = _TRANSITION_BRIDGES["deeper"][0]
        elif energy < 0.3 and user_wants != "play":
            suggested_transition = _TRANSITION_BRIDGES["lighter"][0]

        return TopicTrajectory(
            predicted_next_topic=predicted_next,
            user_likely_wants=user_wants,
            suggested_transition=suggested_transition,
            topics_to_avoid=topics_to_avoid,
            topics_to_surface=topics_to_surface,
        )

    # ── Code-switching ───────────────────────────────────────────────────

    def _compute_code_switch(
        self,
        user_message: str,
        dynamics_state: Any,
        arc: ConversationArc,
    ) -> CodeSwitchContext:
        """Adapt register and vocabulary based on user's style and context."""

        register = getattr(dynamics_state, "register", "casual")
        partner_frame = getattr(dynamics_state, "partner_frame", "neutral")
        partner_intensity = getattr(dynamics_state, "partner_intensity", 0.5)
        msg_lower = user_message.lower()
        words = set(msg_lower.split())

        # ── Emoji mirroring ──
        has_emoji = bool(_EMOJI_PATTERN.search(user_message))

        # ── Slang detection ──
        slang_count = len(words & _SLANG_MARKERS)
        uses_slang = slang_count >= 1

        # ── Vocabulary level ──
        # Simple heuristic: average word length as sophistication proxy
        all_words = [w for w in msg_lower.split() if len(w) > 0]
        avg_word_len = sum(len(w) for w in all_words) / max(len(all_words), 1)

        if avg_word_len > 6.5:
            vocab_level = "sophisticated"
        elif avg_word_len > 5.0:
            vocab_level = "moderate"
        else:
            vocab_level = "simple"

        # Override for technical content
        tech_markers = {"api", "code", "function", "server", "deploy", "debug",
                        "algorithm", "database", "framework", "runtime", "async"}
        if len(words & tech_markers) >= 2:
            vocab_level = "technical"

        # ── Recommended register ──
        # Start from dynamics register, adjust for context
        recommended = register

        # Formality shift for heavy topics
        if partner_frame in ("vulnerable", "serious") and partner_intensity > 0.6:
            if register == "playful":
                recommended = "casual"  # drop the play, stay warm
                formality_reason = "Topic shifted to something serious"
            elif register == "formal":
                recommended = "casual"  # don't be stiff when they're vulnerable
                formality_reason = "Softening formality — they need warmth"
            else:
                formality_reason = None
        elif partner_frame == "playful" and partner_intensity > 0.5:
            recommended = "playful"
            formality_reason = "Energy is up — match the playfulness"
        elif arc.phase == "peak" and partner_frame == "vulnerable":
            recommended = "intimate"
            formality_reason = "Peak vulnerability — be fully present"
        else:
            formality_reason = None

        # Slang users get casual/playful, never formal
        if uses_slang and recommended == "formal":
            recommended = "casual"
            formality_reason = "User uses slang — formal would feel distant"

        return CodeSwitchContext(
            recommended_register=recommended,
            vocabulary_level=vocab_level,
            should_mirror_emoji=has_emoji,
            should_use_slang=uses_slang and recommended in ("casual", "playful", "intimate"),
            formality_shift_reason=formality_reason,
        )

    # ── Public API ───────────────────────────────────────────────────────

    def get_directive(self) -> ConversationalDirective:
        """Return the current conversational directive."""
        return self._directive

    def get_context_injection(self) -> str:
        """
        Format the directive as a context block for system prompt injection.
        Compact, direct, actionable.
        """
        d = self._directive
        arc = d.arc
        pacing = d.pacing
        traj = d.trajectory
        silence = d.silence
        cs = d.code_switch

        lines = ["## CONVERSATIONAL INTELLIGENCE"]

        # ── Arc ──
        energy_desc = self._describe_energy_trend(arc.energy_trajectory)
        peak_note = ""
        if arc.peak_moment and arc.phase in ("peak", "winding_down"):
            peak_note = f" Peaked on: {arc.peak_moment}."
        lines.append(
            f"- **Arc**: {arc.phase.replace('_', ' ').title()} phase "
            f"(turn {arc.total_turns}). {energy_desc}.{peak_note} "
            f"Depth: {arc.depth_reached}."
        )

        # ── Pacing ──
        cadence_desc = {
            "rapid_fire": "They're in rapid-fire mode",
            "conversational": "Normal conversational pace",
            "considered": "Taking their time — be measured",
            "reflective": "Deep reflective mode — you can expand",
        }.get(pacing.cadence, pacing.cadence)

        energy_action = {
            "match_user": "Match their energy",
            "lift_energy": "Gently lift the energy",
            "calm_energy": "Bring calm energy",
            "maintain": "Hold steady",
        }.get(pacing.energy_match, pacing.energy_match)

        lines.append(
            f"- **Pacing**: {energy_action}. "
            f"Keep responses {pacing.target_length} (~{pacing.sentence_count_target} sentences). "
            f"{cadence_desc}."
        )

        # ── Trajectory ──
        wants_desc = {
            "go_deeper": "They want to go deeper on this topic. Don't change subject yet.",
            "change_topic": "Topic is exhausted. Transition naturally.",
            "wrap_up": "They're wrapping up. Keep it brief and warm.",
            "play": "They want to play. Match the energy, be fun.",
            "vent": "They need to vent. Listen, validate, don't fix.",
            "explore": "They're in exploration mode. Follow their lead.",
            "be_quiet": "They may want quiet. Keep it minimal.",
        }.get(traj.user_likely_wants, traj.user_likely_wants)
        lines.append(f"- **Trajectory**: {wants_desc}")

        if traj.topics_to_surface:
            surfaces = ", ".join(traj.topics_to_surface[:3])
            lines.append(f"  Threads to surface: {surfaces}")

        if traj.topics_to_avoid:
            avoids = ", ".join(traj.topics_to_avoid[:3])
            lines.append(f"  Avoid (exhausted): {avoids}")

        # ── Silence ──
        if silence.should_be_quiet:
            reason_desc = {
                "user_processing": "Give them time to process",
                "comfortable_pause": "Comfortable pause — wait for them",
                "user_typing": "They're still typing — hold response",
                "emotional_space": "Give emotional space before responding",
                "topic_exhaustion": "Topic is spent — pause before moving on",
                "natural_end": "Conversation ending naturally",
            }.get(silence.reason, silence.reason)
            wait_note = f" (wait ~{silence.suggested_wait_s:.0f}s)" if silence.suggested_wait_s > 0 else ""
            lines.append(f"- **Silence**: {reason_desc}{wait_note}.")
        elif silence.reason == "emotional_space":
            lines.append(f"- **Silence**: Pause ~{silence.suggested_wait_s:.0f}s — emotional weight. Then respond with care.")
        else:
            lines.append("- **Silence**: Not needed — they're engaged and expecting response.")

        # ── Register ──
        emoji_note = " Emoji OK." if cs.should_mirror_emoji else ""
        slang_note = " Slang OK." if cs.should_use_slang else ""
        shift_note = f" ({cs.formality_shift_reason})" if cs.formality_shift_reason else ""
        lines.append(
            f"- **Register**: {cs.recommended_register.title()}"
            f"{shift_note}. "
            f"Vocab: {cs.vocabulary_level}.{emoji_note}{slang_note}"
        )

        return "\n".join(lines)

    def _describe_energy_trend(self, trajectory: List[float]) -> str:
        """Human-readable description of energy trajectory."""
        if len(trajectory) < 2:
            return "Energy neutral"

        recent = trajectory[-3:] if len(trajectory) >= 3 else trajectory
        avg = sum(recent) / len(recent)

        if len(trajectory) >= 4:
            older = trajectory[-min(len(trajectory), 6):-3]
            if older:
                old_avg = sum(older) / len(older)
                delta = avg - old_avg
                if delta > 0.1:
                    return "Energy rising"
                elif delta < -0.1:
                    return "Energy declining"

        if avg > 0.7:
            return "Energy high"
        elif avg > 0.4:
            return "Energy moderate"
        elif avg > 0.2:
            return "Energy low"
        else:
            return "Energy near zero"


# ── Singleton ───────────────────────────────────────────────────────────────

_engine_instance: Optional[ConversationIntelligenceEngine] = None


def get_conversation_intelligence() -> ConversationIntelligenceEngine:
    """Module-level singleton accessor."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ConversationIntelligenceEngine()
        # Register in ServiceContainer if not already registered
        try:
            from core.container import ServiceContainer
            if not ServiceContainer.has("conversation_intelligence"):
                ServiceContainer.register_instance(
                    "conversation_intelligence", _engine_instance
                )
                logger.info("ConversationIntelligenceEngine registered in ServiceContainer")
        except Exception as e:
            logger.debug("Could not register in ServiceContainer: %s", e)
    return _engine_instance
