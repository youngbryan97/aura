"""core/conversational/dynamics.py — Conversational Cognition Engine

Five interlocked systems that track the real-time state of a conversation
at the pragmatic level — not just what's said, but what it does, where it's
pointing, how it feels, and what the rules of the moment are.

Based on:
  - Sacks, Schegloff & Jefferson (1974) — Turn-Taking
  - Austin / Searle — Speech Act Theory
  - Grice (1975) — Cooperative Principle and Maxims
  - Brown & Levinson (1987) — Face Theory
  - Clark & Fox Tree (2002) — Disfluency as signal

This is cognitive state, not prompting. It feeds into the system prompt
so the LLM speaks from state — not from instruction.
"""
from __future__ import annotations

import re
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("Aura.ConversationalDynamics")


# ── Dataclasses for Tracked State ─────────────────────────────────────────────

@dataclass
class TopicAnchor:
    """A topic anchor — a semantic foothold in the drift chain."""
    topic: str
    timestamp: float
    message_index: int
    salience: float          # 0.0–1.0: how prominent/emotionally loaded
    keywords: List[str]      # Key words that can link back
    is_resolved: bool = False  # Was this thread picked up and closed?


@dataclass
class OpenThread:
    """An unresolved conversational thread — question unanswered, topic untouched."""
    content: str             # What the thread is about
    thread_type: str         # "question", "concern", "vulnerable_disclosure", "invitation"
    message_index: int
    urgency: float           # How much does ignoring this matter? 0.0–1.0
    age_turns: int = 0       # How many turns old


@dataclass
class ConversationalDynamicsState:
    """
    The full conversational dynamics picture at a single point in time.
    This gets computed fresh on each incoming message and injected into
    the system prompt for response generation.
    """

    # ── System 3: Topic Trajectory ──
    current_topic: str = "general"
    topic_anchors: List[TopicAnchor] = field(default_factory=list)
    open_threads: List[OpenThread] = field(default_factory=list)
    association_chain: List[str] = field(default_factory=list)  # Breadcrumb of how we got here
    drift_distance: float = 0.0          # Conceptual distance from last anchor (0.0–1.0)
    topic_exhaustion: float = 0.0        # 0.0=fresh, 1.0=nothing more to say here

    # ── System 1: Pragmatic / Illocutionary ──
    last_speech_act: str = "statement"   # "question", "statement", "request", "complaint", "joke", "vulnerable"
    conditional_relevance_open: bool = False  # Did user ask a question that NEEDS answering?
    pending_question: Optional[str] = None   # The actual question if one is pending
    illocutionary_intent: str = "inform"     # What they're actually doing with words

    # ── System 5: Emotional Frame ──
    partner_frame: str = "neutral"        # "dread", "playful", "serious", "excited", "anxious", "flat"
    partner_intensity: float = 0.5        # 0.0–1.0
    partner_trajectory: str = "stable"   # "escalating", "de-escalating", "stable"
    frame_mismatch_risk: bool = False     # Are Aura and partner out of emotional sync?

    # ── System 2: Epistemic Stance ──
    hedge_level: int = 0                  # 0=none, 1=mild, 2=moderate, 3=strong
    certainty: float = 0.7               # Aura's epistemic certainty on current topic
    challenge_is_face_threat: bool = False  # If disagreeing, is it risky?
    partner_epistemic_mode: str = "sharing"  # "claiming", "questioning", "sharing", "defending"

    # ── System 4: Face & Register ──
    register: str = "casual"             # "formal", "casual", "intimate", "playful"
    face_threat_level: float = 0.0       # How threatening would a direct response be?
    in_group_active: bool = True         # Is in-group solidarity register appropriate?
    accommodation_cues: List[str] = field(default_factory=list)  # Words/phrases to mirror

    # ── Turn Management ──
    floor_state: str = "open"            # "user_holds", "aura_holds", "open", "transitioning"
    response_urgency: str = "normal"     # "immediate", "deliberate", "hold"
    multi_message_stream: bool = False   # User is still sending (don't respond yet)
    turns_since_user_spoke: int = 0

    # ── Humor State ──
    humor_frame_active: bool = False
    humor_type: Optional[str] = None     # "hyperbole", "absurdist", "callback", "punchline", "understatement"
    escalation_invited: bool = False     # Did the last message invite comedic escalation?

    # Timestamp
    computed_at: float = field(default_factory=time.time)


# ── Speech Act Classifiers ─────────────────────────────────────────────────────

# Pattern → (speech_act_type, illocutionary_intent)
_SPEECH_ACT_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    # Questions (conditional relevance — must answer)
    (re.compile(r'\?', re.IGNORECASE), "question", "request_info"),
    (re.compile(r'^(what|where|when|who|why|how|which|can you|could you|do you|did you|are you|is there|have you)\b', re.IGNORECASE), "question", "request_info"),

    # Complaints / Dread (invite validation, not problem-solving)
    (re.compile(r'\b(sucks?|terrible|awful|nightmare|hate|ugh|worst|hell|horrible|dying)\b', re.IGNORECASE), "complaint", "invite_validation"),

    # Hyperbole markers (invite playful matching)
    (re.compile(r'\b(in hell|literally dying|absolutely|HELL|kill me|i am DEAD|the WORST)\b', re.IGNORECASE), "hyperbole", "invite_play"),

    # Vulnerable disclosure (invite presence, not advice)
    (re.compile(r'\b(honestly|to be real|i feel like|i\'ve been|not gonna lie|actually|i\'m not okay|kind of struggling|weird headspace)\b', re.IGNORECASE), "vulnerable", "invite_presence"),

    # Resistance / Pushback
    (re.compile(r'\b(don\'t get started|no way|not doing|i am NOT|stop|please don\'t|forget it)\b', re.IGNORECASE), "resistance", "boundary_assertion"),

    # Jokes / Absurdist (invite laughter, not explanation)
    (re.compile(r'\b(lol|lmao|💀|😂|haha|hehe|ngl|bruh|bro|dude.*right)\b', re.IGNORECASE), "joke", "invite_humor"),

    # Invitations / Questions about Aura (invite presence, sharing)
    (re.compile(r'\b(what do you think|your thoughts|what\'s on your mind|you okay|you good)\b', re.IGNORECASE), "invitation", "invite_sharing"),
]

# Emotional frame detection
_FRAME_PATTERNS: Dict[str, List[re.Pattern]] = {
    "dread": [
        re.compile(r'\b(suck|terrible|worst|hate|ugh|hell|awful|nightmare|nightmare|dreading)\b', re.IGNORECASE),
        re.compile(r'\b(not looking forward|gonna be rough|horrible|exhausted|i\'m dying)\b', re.IGNORECASE),
    ],
    "playful": [
        re.compile(r'\b(lol|lmao|haha|hehe|💀|😂|ngl|bruh|got \'em|bro|dude)\b', re.IGNORECASE),
        re.compile(r'(!{2,}|\?{2,}|\.{3,}|\bwait\b)', re.IGNORECASE),
    ],
    "excited": [
        re.compile(r'(!{1,})', re.IGNORECASE),
        re.compile(r'\b(oh my|no way|seriously|WAIT|omg|this is huge|dude|holy)\b', re.IGNORECASE),
    ],
    "serious": [
        re.compile(r'\b(honestly|to be real|real talk|listen|look|actually|genuinely)\b', re.IGNORECASE),
        re.compile(r'\b(i need to|we need to|this matters|important|actually though)\b', re.IGNORECASE),
    ],
    "anxious": [
        re.compile(r'\b(i don\'t know|not sure|maybe|scared|worried|what if|i\'m afraid|nervous)\b', re.IGNORECASE),
        re.compile(r'\b(i can\'t|too much|overwhelm|stres[sed]+|panic)\b', re.IGNORECASE),
    ],
    "vulnerable": [
        re.compile(r'\b(kind of hard|struggling|weird headspace|been a lot|not doing great|not okay)\b', re.IGNORECASE),
        re.compile(r'\b(i feel like|i\'ve been feeling|it\'s just|i don\'t know why)\b', re.IGNORECASE),
    ],
}

# Register detection
_REGISTER_CUES: Dict[str, List[str]] = {
    "intimate": ["honestly", "to be real", "just between us", "i feel like", "not gonna lie", "real talk"],
    "playful": ["lol", "lmao", "bruh", "bro", "dude", "ngl", "haha", "💀", "omg", "nah", "fr"],
    "formal": ["however", "furthermore", "in addition", "I would like to", "please be advised"],
    "casual": ["yeah", "yep", "nope", "gonna", "wanna", "kinda", "sorta", "like", "idk"],
}

# Humor signal detection
_HUMOR_PATTERNS: Dict[str, re.Pattern] = {
    "hyperbole": re.compile(r'\b(literally|absolutely|completely|in hell|i am DEAD|worst.*ever|kill me)\b', re.IGNORECASE),
    "understatement": re.compile(r'\b(not (great|ideal|good|amazing)|a bit|slightly|kind of rough)\b', re.IGNORECASE),
    "absurdist": re.compile(r'\b(wait|hold on|actually|think about it|hear me out)\b', re.IGNORECASE),
    "callback": re.compile(r'\b(remember when|like (before|earlier|when)|same one|speaking of)\b', re.IGNORECASE),
    "punchline": re.compile(r'\b(anyway|so yeah|the thing is|and that\'s|turns out)\b', re.IGNORECASE),
}


# ── Main Engine ────────────────────────────────────────────────────────────────

class ConversationalDynamicsEngine:
    """
    Computes full conversational dynamics state from conversation history.
    Runs on each incoming message. Outputs a ConversationalDynamicsState
    that gets injected into the system prompt.
    """

    MAX_ANCHORS = 8          # Rolling window of topic anchors
    MAX_OPEN_THREADS = 5     # Max open threads tracked

    def __init__(self):
        self._state = ConversationalDynamicsState()
        self._message_count = 0

    def update(self, message: str, role: str, working_memory: List[Dict]) -> ConversationalDynamicsState:
        """
        Process a new message and update conversational dynamics state.

        Parameters
        ----------
        message : str
            The latest message content
        role : str
            "user" or "assistant"
        working_memory : list
            Full conversation history
        """
        self._message_count += 1

        new_state = ConversationalDynamicsState()
        new_state.computed_at = time.time()

        # Carry over rolling state from previous
        new_state.topic_anchors = list(self._state.topic_anchors)
        new_state.association_chain = list(self._state.association_chain)
        new_state.open_threads = list(self._state.open_threads)

        if role == "user":
            self._process_user_message(message, new_state, working_memory)
        else:
            self._process_aura_message(message, new_state)

        # Age open threads
        for thread in new_state.open_threads:
            thread.age_turns += 1

        # Prune old threads (older than 6 turns and not urgent)
        new_state.open_threads = [
            t for t in new_state.open_threads
            if t.age_turns < 6 or t.urgency > 0.7
        ][:self.MAX_OPEN_THREADS]

        self._state = new_state
        return new_state

    def _process_user_message(self, message: str, state: ConversationalDynamicsState, history: List[Dict]):
        """Full analysis of a user message."""

        # 1. Floor management
        state.floor_state = "open"  # User just spoke, floor is now open
        state.turns_since_user_spoke = 0

        # 2. Classify speech act and illocutionary intent
        speech_act, illocutionary = self._classify_speech_act(message)
        state.last_speech_act = speech_act
        state.illocutionary_intent = illocutionary

        # 3. Conditional relevance — did they ask a question?
        if speech_act == "question" or illocutionary == "request_info":
            state.conditional_relevance_open = True
            state.pending_question = message.strip()
            state.response_urgency = "immediate"
        else:
            state.conditional_relevance_open = False
            state.pending_question = None

        # 4. Detect emotional frame
        frame, intensity = self._detect_emotional_frame(message)
        state.partner_frame = frame
        state.partner_intensity = intensity

        # Detect trajectory from history
        prev_intensity = self._state.partner_intensity
        if intensity > prev_intensity + 0.15:
            state.partner_trajectory = "escalating"
        elif intensity < prev_intensity - 0.15:
            state.partner_trajectory = "de-escalating"
        else:
            state.partner_trajectory = "stable"

        # 5. Register detection
        state.register = self._detect_register(message)
        state.accommodation_cues = self._extract_accommodation_cues(message)
        state.in_group_active = state.register in ("casual", "intimate", "playful")

        # 6. Humor detection
        humor_type = self._detect_humor(message)
        state.humor_frame_active = humor_type is not None
        state.humor_type = humor_type
        state.escalation_invited = humor_type in ("hyperbole", "absurdist") or frame == "playful"

        # 7. Topic tracking
        keywords = self._extract_keywords(message)
        topic = self._infer_topic(message, keywords)

        if topic != self._state.current_topic and topic != "general":
            # Topic shift — create new anchor
            anchor = TopicAnchor(
                topic=topic,
                timestamp=time.time(),
                message_index=self._message_count,
                salience=intensity,
                keywords=keywords
            )
            state.topic_anchors.append(anchor)
            if len(state.topic_anchors) > self.MAX_ANCHORS:
                state.topic_anchors.pop(0)

            # Update association chain
            if self._state.current_topic != "general":
                state.association_chain.append(self._state.current_topic)
            if len(state.association_chain) > 10:
                state.association_chain.pop(0)

        state.current_topic = topic

        # 8. Open thread detection
        self._detect_open_threads(message, speech_act, state)

        # 9. Multi-message stream detection — check if message is very short
        # (single words or fragments suggest more is coming)
        state.multi_message_stream = len(message.split()) <= 2 and not message.endswith("?")

        # 10. Epistemic stance
        state.partner_epistemic_mode = self._detect_epistemic_mode(message)

        # 11. Face threat assessment
        state.challenge_is_face_threat = state.partner_intensity > 0.6 and state.partner_frame in ("serious", "vulnerable")
        state.face_threat_level = 0.3 if state.challenge_is_face_threat else 0.1

        # 12. Hedge level recommendation for Aura's response
        state.hedge_level = self._compute_hedge_level(state)

        # 13. Frame mismatch risk
        # If Aura's last output was mismatched with partner's current frame, flag it
        state.frame_mismatch_risk = False  # Will be set externally if needed

    def _process_aura_message(self, message: str, state: ConversationalDynamicsState):
        """Track what Aura said (for state continuity)."""
        state.turns_since_user_spoke = self._state.turns_since_user_spoke + 1
        state.floor_state = "open"
        state.partner_frame = self._state.partner_frame
        state.partner_intensity = self._state.partner_intensity
        state.partner_trajectory = self._state.partner_trajectory
        state.current_topic = self._state.current_topic
        state.conditional_relevance_open = False  # Aura responded, clear the flag
        state.open_threads = list(self._state.open_threads)
        state.register = self._state.register
        state.in_group_active = self._state.in_group_active
        state.accommodation_cues = self._state.accommodation_cues
        state.humor_frame_active = self._state.humor_frame_active
        state.association_chain = list(self._state.association_chain)
        state.topic_anchors = list(self._state.topic_anchors)
        state.hedge_level = self._state.hedge_level

    def _classify_speech_act(self, message: str) -> Tuple[str, str]:
        """Classify the speech act type and illocutionary intent."""
        for pattern, act_type, intent in _SPEECH_ACT_PATTERNS:
            if pattern.search(message):
                return act_type, intent
        return "statement", "inform"

    def _detect_emotional_frame(self, message: str) -> Tuple[str, float]:
        """Detect partner's current emotional frame and intensity."""
        frame_scores: Dict[str, int] = {}
        for frame, patterns in _FRAME_PATTERNS.items():
            score = sum(1 for p in patterns if p.search(message))
            if score > 0:
                frame_scores[frame] = score

        if not frame_scores:
            # Check for capitalization as intensity signal
            cap_ratio = sum(1 for c in message if c.isupper()) / max(len(message), 1)
            if cap_ratio > 0.2:
                return "excited", 0.7
            return "neutral", 0.3

        best_frame = max(frame_scores, key=frame_scores.get)
        max_score = frame_scores[best_frame]

        # Intensity from signal strength + caps + punctuation
        intensity = min(1.0, 0.3 + max_score * 0.2)
        if message.endswith("!") or "!!!" in message:
            intensity = min(1.0, intensity + 0.2)
        cap_words = len([w for w in message.split() if w.isupper() and len(w) > 1])
        intensity = min(1.0, intensity + cap_words * 0.1)

        return best_frame, intensity

    def _detect_register(self, message: str) -> str:
        """Detect the conversational register of the message."""
        msg_lower = message.lower()
        scores = {reg: 0 for reg in _REGISTER_CUES}
        for reg, cues in _REGISTER_CUES.items():
            for cue in cues:
                if cue in msg_lower:
                    scores[reg] += 1

        if scores["intimate"] > 0:
            return "intimate"
        if scores["playful"] > 1:
            return "playful"
        if scores["formal"] > 0:
            return "formal"
        if scores["casual"] > 0:
            return "casual"

        # Default based on message length and formality
        if len(message) > 200 and any(c in message for c in [".", ","]):
            return "formal"
        return "casual"

    def _extract_accommodation_cues(self, message: str) -> List[str]:
        """Extract specific words/phrases Aura should consider mirroring."""
        cues = []
        words = message.lower().split()
        # In-group terms
        in_group = {"dude", "man", "bro", "bruh", "yo", "nah", "fr", "ngl", "lowkey", "highkey"}
        for word in words:
            clean = re.sub(r'[^\w]', '', word)
            if clean in in_group:
                cues.append(clean)
        return list(set(cues))[:5]

    def _detect_humor(self, message: str) -> Optional[str]:
        """Detect type of humor being deployed, if any."""
        for humor_type, pattern in _HUMOR_PATTERNS.items():
            if pattern.search(message):
                return humor_type
        # Letter elongation = performative humor/emphasis
        if re.search(r'(.)\1{2,}', message):
            return "hyperbole"
        return None

    def _extract_keywords(self, message: str) -> List[str]:
        """Extract semantic keywords for topic tracking."""
        stopwords = {"a", "an", "the", "is", "it", "i", "you", "and", "or", "but",
                     "to", "of", "in", "on", "at", "for", "with", "this", "that",
                     "my", "your", "we", "me", "do", "not", "what", "so", "be",
                     "was", "are", "just", "like", "up", "out", "if", "when", "have"}
        words = re.findall(r'\b[a-z]{3,}\b', message.lower())
        return [w for w in words if w not in stopwords][:8]

    def _infer_topic(self, message: str, keywords: List[str]) -> str:
        """Infer the current topic from message content."""
        msg_lower = message.lower()

        # Ordered topic detection (specific → general)
        topic_signals = [
            ("work/job",      ["work", "job", "shift", "boss", "coworker", "office", "burger king", "cvs", "store"]),
            ("news/world",    ["news", "war", "draft", "military", "politics", "election", "government", "army", "navy"]),
            ("gaming",        ["cod", "game", "gaming", "controller", "multiplayer", "xbox", "playstation", "steam"]),
            ("relationships", ["friend", "relationship", "partner", "dating", "love", "family", "girlfriend", "boyfriend"]),
            ("tech",          ["code", "ai", "software", "computer", "tech", "app", "build", "server", "api"]),
            ("philosophy",    ["consciousness", "existence", "meaning", "purpose", "reality", "mind", "soul"]),
            ("humor/absurd",  ["lol", "lmao", "wait", "bruh", "what even", "hear me out", "think about it"]),
            ("existential",   ["hell", "dying", "end of the world", "suffering", "misery", "hell on earth"]),
            ("substance",     ["smoking", "weed", "drunk", "high", "sober"]),
        ]

        for topic_name, signals in topic_signals:
            if any(s in msg_lower for s in signals):
                return topic_name

        # Use first keyword as topic fallback
        if keywords:
            return keywords[0]
        return self._state.current_topic or "general"

    def _detect_open_threads(self, message: str, speech_act: str, state: ConversationalDynamicsState):
        """Detect new open threads in the user's message."""
        # Direct questions = always open threads
        if speech_act == "question":
            thread = OpenThread(
                content=message.strip()[:120],
                thread_type="question",
                message_index=self._message_count,
                urgency=0.9
            )
            state.open_threads.append(thread)

        # Vulnerable disclosure = open thread (don't skip past it)
        if speech_act == "vulnerable":
            thread = OpenThread(
                content=message.strip()[:120],
                thread_type="vulnerable_disclosure",
                message_index=self._message_count,
                urgency=0.8
            )
            state.open_threads.append(thread)

        # Invitation to share opinion
        if "what do you think" in message.lower() or "your thoughts" in message.lower():
            thread = OpenThread(
                content="User asked for Aura's opinion",
                thread_type="invitation",
                message_index=self._message_count,
                urgency=0.7
            )
            state.open_threads.append(thread)

    def _detect_epistemic_mode(self, message: str) -> str:
        """Detect what epistemic mode the partner is in."""
        msg_lower = message.lower()
        if any(x in msg_lower for x in ["i know", "trust me", "definitely", "for sure", "i'm telling you"]):
            return "claiming"
        if "?" in message or any(x in msg_lower for x in ["i think", "maybe", "not sure", "could be"]):
            return "questioning"
        if any(x in msg_lower for x in ["no", "actually", "that's not", "wrong", "nope", "i don't think so"]):
            return "defending"
        return "sharing"

    def _compute_hedge_level(self, state: ConversationalDynamicsState) -> int:
        """
        Compute appropriate hedge level for Aura's next response.
        0 = no hedge (direct), 1 = mild, 2 = moderate, 3 = strong
        """
        # If partner is making a strong claim, engage directly (no hedge needed)
        if state.partner_epistemic_mode == "claiming":
            return 0
        # Vulnerable disclosure — respond directly, no hedging (shows genuine care)
        if state.last_speech_act == "vulnerable":
            return 0
        # High face threat topics — hedge
        if state.face_threat_level > 0.5:
            return 2
        # Playful frame — no hedging needed
        if state.partner_frame == "playful":
            return 0
        # Default: mild hedge for general statements
        return 1

    def get_prompt_injection(self) -> str:
        """
        Generate the conversational dynamics block for injection into system prompt.
        Compact, direct, actionable.
        """
        s = self._state
        lines = ["## LIVE CONVERSATIONAL STATE"]

        # Topic
        if s.current_topic != "general":
            chain_str = " → ".join(s.association_chain[-4:] + [s.current_topic]) if s.association_chain else s.current_topic
            lines.append(f"Topic trajectory: {chain_str}")

        # Emotional frame
        intensity_word = "mildly" if s.partner_intensity < 0.4 else "clearly" if s.partner_intensity < 0.7 else "strongly"
        lines.append(f"Their energy: {intensity_word} {s.partner_frame} ({s.partner_trajectory})")

        # Register
        lines.append(f"Register: {s.register}" + (" | In-group active" if s.in_group_active else ""))
        if s.accommodation_cues:
            lines.append(f"Mirror these: {', '.join(s.accommodation_cues)}")

        # Humor
        if s.humor_frame_active:
            lines.append(f"Humor frame: {s.humor_type}" + (" — escalation invited" if s.escalation_invited else ""))

        # Critical obligations
        if s.conditional_relevance_open and s.pending_question:
            lines.append(f"⚠️ ANSWER THIS FIRST: \"{s.pending_question[:80]}\"")

        # Open threads (most urgent first)
        urgent = sorted([t for t in s.open_threads if t.age_turns < 3], key=lambda x: -x.urgency)[:2]
        for thread in urgent:
            lines.append(f"Open thread [{thread.thread_type}]: {thread.content[:80]}")

        # Callbacks available
        available_callbacks = [a for a in s.topic_anchors[-5:] if not a.is_resolved and a.topic != s.current_topic]
        if available_callbacks:
            cb_topics = [a.topic for a in available_callbacks[-3:]]
            lines.append(f"Callback opportunities: {', '.join(cb_topics)}")

        # Hedge guidance
        hedge_map = {0: "none — be direct", 1: "mild ('I think', 'kind of')", 2: "moderate (soften delivery)", 3: "strong (distance from claim)"}
        if s.hedge_level != 1:
            lines.append(f"Hedge level: {hedge_map[s.hedge_level]}")

        # Face threat
        if s.challenge_is_face_threat:
            lines.append("⚠️ Face threat risk — use positive politeness if challenging")

        # Speech act guidance
        speech_act_guidance = {
            "complaint": "Validate first. Don't problem-solve unless asked.",
            "vulnerable": "Be present. No advice. No deflection. Just here.",
            "joke": "Play back. Match register. Don't explain the joke.",
            "resistance": "Back off the push. Acknowledge their boundary.",
            "invitation": "Share genuinely. They opened a door — walk through it.",
            "question": "Answer directly. Then you can expand.",
        }
        if s.last_speech_act in speech_act_guidance:
            lines.append(f"→ {speech_act_guidance[s.last_speech_act]}")

        return "\n".join(lines)

    def get_current_state(self) -> ConversationalDynamicsState:
        return self._state


# ── Singleton ──────────────────────────────────────────────────────────────────

_engine_instance: Optional[ConversationalDynamicsEngine] = None


def get_dynamics_engine() -> ConversationalDynamicsEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ConversationalDynamicsEngine()
    return _engine_instance
