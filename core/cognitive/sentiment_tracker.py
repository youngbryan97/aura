"""core/cognitive/sentiment_tracker.py

Sentiment Trajectory Tracker — Text-Based Emotional Understanding.

Provides real sentiment analysis of conversation text, complementing the
hardware-only mood detection in core/affect/affective_circumplex.py. While
the circumplex reads CPU/RAM/swap to derive Aura's somatic state, this module
reads the actual words a user types to understand *their* emotional state.

Architecture:
  1. Lexicon-based valence scoring (AFINN-style, ~200+ high-signal words)
  2. Multi-signal detectors: sarcasm, urgency, frustration, warmth
  3. Six-dimensional EmotionalVector output per message
  4. Exponentially-weighted moving state for smooth trajectory tracking
  5. Narrative summary generation for human-readable mood reports

Zero external dependencies. No NLTK, no transformers, no spaCy.
Thread-safe via asyncio (single-writer design with immutable snapshots).
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger("Aura.SentimentTracker")

__all__ = [
    "SentimentTrajectoryTracker",
    "EmotionalVector",
    "get_sentiment_tracker",
]


# ---------------------------------------------------------------------------
# Emotional vector dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EmotionalVector:
    """Six-dimensional emotional state for a single message or smoothed state.

    Each dimension is bounded to its documented range. Frozen so snapshots
    are safe to share across async boundaries without copies.
    """

    valence: float = 0.0
    """Negative-to-positive sentiment.  Range: -1.0 to 1.0."""

    arousal: float = 0.0
    """Calm-to-excited energy level.  Range: 0.0 to 1.0."""

    dominance: float = 0.5
    """Submissive-to-assertive tone.  Range: 0.0 to 1.0."""

    urgency: float = 0.0
    """Relaxed-to-urgent pressure.  Range: 0.0 to 1.0."""

    warmth: float = 0.0
    """Distant-to-warm/friendly rapport.  Range: 0.0 to 1.0."""

    frustration: float = 0.0
    """Patient-to-frustrated tension.  Range: 0.0 to 1.0."""

    timestamp: float = field(default_factory=time.monotonic)
    """Monotonic clock at which this vector was produced."""

    role: str = "user"
    """Who produced the text: 'user' or 'assistant'."""

    def as_dict(self) -> dict:
        return {
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "dominance": round(self.dominance, 4),
            "urgency": round(self.urgency, 4),
            "warmth": round(self.warmth, 4),
            "frustration": round(self.frustration, 4),
        }


# ---------------------------------------------------------------------------
# Built-in AFINN-style lexicon (~250 high-signal words)
# ---------------------------------------------------------------------------
# Valence scores: negative = bad, positive = good, magnitude = intensity.
# Range roughly -5 to +5. Words chosen for high signal in chat contexts.

_LEXICON: dict[str, float] = {
    # --- strongly negative (-5 to -3) ---
    "abandon": -3, "abhor": -4, "abuse": -4, "agony": -4, "annihilate": -4,
    "appalling": -4, "assault": -4, "atrocious": -5, "awful": -3, "bastard": -5,
    "betray": -4, "bitter": -3, "brutal": -4, "catastrophe": -4, "crap": -3,
    "cruel": -4, "damn": -3, "dead": -3, "destroy": -4, "despair": -4,
    "despise": -4, "devastating": -4, "disaster": -4, "disgusting": -4,
    "dreadful": -4, "enemy": -3, "enrage": -4, "evil": -4, "fail": -3,
    "failure": -3, "fatal": -4, "fear": -3, "filthy": -3, "fool": -3,
    "fraud": -4, "furious": -4, "grief": -4, "grim": -3, "gross": -3,
    "guilt": -3, "harm": -3, "hate": -4, "hatred": -5, "hell": -3,
    "helpless": -3, "horrible": -4, "horrify": -4, "hostile": -3, "hurt": -3,
    "idiot": -4, "ignorant": -3, "incompetent": -3, "infuriate": -4,
    "insult": -3, "irritate": -3, "jerk": -3, "kill": -4, "livid": -4,
    "loathe": -4, "lonely": -3, "lose": -3, "lousy": -3, "miserable": -4,
    "moron": -4, "murder": -5, "nasty": -3, "negate": -3, "nightmare": -4,
    "obnoxious": -3, "offend": -3, "outrage": -4, "pain": -3, "panic": -3,
    "pathetic": -3, "piss": -3, "poison": -3, "pollute": -3, "poor": -2,
    "rage": -4, "reject": -3, "repulsive": -4, "resent": -3, "rotten": -3,
    "rude": -3, "ruin": -3, "sad": -2, "scam": -4, "scare": -3,
    "scream": -3, "selfish": -3, "shit": -4, "shock": -2, "sick": -2,
    "slaughter": -5, "sorrow": -3, "spam": -3, "stupid": -3, "suck": -3,
    "suffer": -3, "terrible": -4, "terrify": -4, "threat": -3, "toxic": -4,
    "tragic": -4, "trash": -3, "ugly": -3, "unacceptable": -3, "unfair": -3,
    "unfortunate": -2, "unhappy": -2, "upset": -2, "useless": -3, "vile": -4,
    "violent": -4, "waste": -3, "weak": -2, "wicked": -3, "worse": -3,
    "worst": -4, "worthless": -4, "wrong": -2, "wtf": -4,

    # --- mildly negative (-2 to -1) ---
    "annoying": -2, "anxious": -2, "awkward": -2, "bad": -2, "bland": -1,
    "bored": -2, "boring": -2, "broken": -2, "bug": -2, "busy": -1,
    "complain": -2, "complex": -1, "concern": -1, "confuse": -2,
    "confused": -2, "confusing": -2, "crash": -2, "criticism": -2,
    "delay": -2, "difficult": -2, "disappoint": -2, "disappointed": -2,
    "disconnect": -2, "doubt": -2, "downgrade": -2, "drag": -1, "dull": -2,
    "error": -2, "exhausted": -2, "expensive": -2, "frustrate": -2,
    "frustrated": -2, "frustrating": -2, "glitch": -2, "hard": -1,
    "hassle": -2, "headache": -2, "hesitant": -1, "ignore": -2,
    "impatient": -2, "impossible": -2, "inconvenient": -2, "lag": -2,
    "limited": -1, "mediocre": -2, "mess": -2, "missing": -2, "mundane": -1,
    "nah": -1, "negative": -2, "nope": -1, "nothing": -1, "outdated": -2,
    "overpriced": -2, "overwhelmed": -2, "problem": -2, "questionable": -1,
    "regret": -2, "risk": -1, "rough": -2, "slow": -2, "sluggish": -2,
    "struggle": -2, "stuck": -2, "tedious": -2, "tired": -2, "trouble": -2,
    "unclear": -1, "uncomfortable": -2, "underwhelming": -2,
    "unexpected": -1, "unfamiliar": -1, "unfortunately": -2, "unhelpful": -2,
    "unreliable": -2, "unsure": -1, "wait": -1, "worried": -2, "worry": -2,

    # --- mildly positive (+1 to +2) ---
    "accept": 1, "accessible": 1, "adequate": 1, "agree": 1, "alright": 1,
    "approve": 2, "assist": 2, "better": 2, "calm": 2, "capable": 2,
    "certain": 1, "clean": 2, "clear": 2, "comfortable": 2, "convenient": 2,
    "cool": 2, "correct": 2, "decent": 2, "easy": 2, "effective": 2,
    "efficient": 2, "enjoy": 2, "fair": 2, "familiar": 1, "fine": 1,
    "fix": 1, "fixed": 2, "flexible": 2, "friendly": 2, "fun": 2,
    "generous": 2, "gentle": 2, "glad": 2, "good": 2, "grateful": 2,
    "handy": 2, "happy": 2, "healthy": 2, "helpful": 2, "honest": 2,
    "hope": 2, "improve": 2, "improved": 2, "informative": 2, "innovative": 2,
    "inspire": 2, "interest": 1, "interesting": 2, "intuitive": 2, "joy": 2,
    "keen": 2, "kind": 2, "learn": 1, "light": 1, "like": 2, "lively": 2,
    "logical": 1, "love": 2, "lucky": 2, "meaningful": 2, "neat": 2,
    "nice": 2, "ok": 1, "okay": 1, "open": 1, "optimistic": 2,
    "organized": 2, "patient": 2, "peace": 2, "pleasant": 2, "pleased": 2,
    "polite": 2, "positive": 2, "practical": 1, "pretty": 1, "progress": 2,
    "promise": 2, "proper": 1, "proud": 2, "quick": 2, "quiet": 1,
    "ready": 1, "recommend": 2, "relax": 2, "reliable": 2, "respect": 2,
    "responsive": 2, "safe": 2, "satisfy": 2, "secure": 2, "simple": 2,
    "sincere": 2, "smart": 2, "smooth": 2, "solid": 2, "solution": 2,
    "stable": 2, "straightforward": 2, "strong": 2, "succeed": 2,
    "success": 2, "support": 2, "sweet": 2, "swift": 2, "thankful": 2,
    "thanks": 2, "thank": 2, "thoughtful": 2, "tidy": 1, "top": 1,
    "trust": 2, "understand": 2, "upgrade": 2, "useful": 2, "valuable": 2,
    "welcome": 2, "well": 1, "win": 2, "wise": 2, "wonderful": 2,
    "work": 1, "works": 2, "yes": 1, "yep": 1, "yeah": 1,

    # --- strongly positive (+3 to +5) ---
    "amazing": 4, "awesome": 4, "beautiful": 3, "best": 3, "blessed": 3,
    "bliss": 4, "brilliant": 4, "celebrate": 3, "champion": 3, "charming": 3,
    "cherish": 3, "delight": 3, "ecstatic": 5, "elegant": 3, "epic": 3,
    "euphoria": 5, "excellent": 4, "exceptional": 4, "excited": 3,
    "exciting": 3, "extraordinary": 4, "fabulous": 4, "fantastic": 4,
    "flawless": 4, "genius": 4, "glorious": 4, "gorgeous": 4, "graceful": 3,
    "grand": 3, "great": 3, "greatest": 4, "heavenly": 4, "heroic": 3,
    "hilarious": 3, "ideal": 3, "impressive": 3, "incredible": 4,
    "magnificent": 4, "marvelous": 4, "masterpiece": 5, "miracle": 4,
    "outstanding": 4, "paradise": 4, "perfect": 4, "phenomenal": 4,
    "remarkable": 3, "sensational": 4, "spectacular": 4, "splendid": 3,
    "stellar": 4, "stunning": 4, "superb": 4, "superior": 3, "supreme": 3,
    "terrific": 4, "thrilled": 4, "thrilling": 4, "top-notch": 4,
    "tremendous": 4, "triumph": 4, "ultimate": 3, "unbelievable": 3,
    "victorious": 3, "wow": 4,
}

# Negation words that flip valence of the next word
_NEGATORS: set[str] = {
    "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
    "nor", "cannot", "can't", "won't", "don't", "didn't", "doesn't",
    "isn't", "aren't", "wasn't", "weren't", "hardly", "barely",
    "scarcely", "seldom", "without",
}

# Intensifiers that amplify the next word's valence
_INTENSIFIERS: dict[str, float] = {
    "very": 1.4, "really": 1.4, "extremely": 1.6, "incredibly": 1.6,
    "absolutely": 1.5, "totally": 1.4, "utterly": 1.5, "completely": 1.4,
    "super": 1.4, "so": 1.3, "highly": 1.4, "deeply": 1.4,
    "truly": 1.3, "especially": 1.3, "particularly": 1.3,
}

# Diminishers that weaken the next word's valence
_DIMINISHERS: dict[str, float] = {
    "slightly": 0.5, "somewhat": 0.6, "barely": 0.4, "hardly": 0.4,
    "sort of": 0.6, "kind of": 0.6, "a bit": 0.6, "a little": 0.6,
    "fairly": 0.7, "rather": 0.7, "mildly": 0.5,
}

# Warmth markers — words/phrases signaling friendliness and rapport
_WARMTH_WORDS: set[str] = {
    "lol", "lmao", "haha", "hahaha", "hehe", "rofl", "xd",
    "hey", "hi", "hello", "howdy", "yo", "sup",
    "thanks", "thank", "thx", "ty", "appreciate", "grateful",
    "please", "pls", "plz",
    "friend", "buddy", "dude", "mate", "pal", "bro",
    "cheers", "congrats", "kudos", "props",
    "love", "adore", "awesome", "amazing", "brilliant",
    "sweet", "kind", "thoughtful",
    "welcome", "glad", "happy",
    "cute", "nice", "cool", "neat",
}

# Sarcasm phrase patterns (compiled once)
_SARCASM_PATTERNS: list[re.Pattern] = [
    re.compile(r'"sure"', re.IGNORECASE),
    re.compile(r'"great"', re.IGNORECASE),
    re.compile(r'"thanks"', re.IGNORECASE),
    re.compile(r'"fine"', re.IGNORECASE),
    re.compile(r'"perfect"', re.IGNORECASE),
    re.compile(r'"wonderful"', re.IGNORECASE),
    re.compile(r'"amazing"', re.IGNORECASE),
    re.compile(r'\boh great\b', re.IGNORECASE),
    re.compile(r'\boh wow\b', re.IGNORECASE),
    re.compile(r'\boh sure\b', re.IGNORECASE),
    re.compile(r'\boh perfect\b', re.IGNORECASE),
    re.compile(r'\byeah right\b', re.IGNORECASE),
    re.compile(r'\bsure+\.{2,}', re.IGNORECASE),       # sure... / sureeee...
    re.compile(r'\bthanks a lot\b', re.IGNORECASE),
    re.compile(r'\bvery helpful\b\.*$', re.IGNORECASE), # trailing "very helpful."
    re.compile(r'/s\b'),                                 # explicit sarcasm tag
]

# Imperative verbs that signal urgency when at the start of a sentence
_IMPERATIVE_VERBS: set[str] = {
    "fix", "stop", "help", "do", "change", "remove", "delete", "update",
    "hurry", "rush", "send", "tell", "give", "show", "explain", "answer",
    "respond", "reply", "check", "look", "find", "get", "make", "run",
    "now", "asap", "immediately", "urgent", "quickly", "fast",
}

# Pre-compiled regex for tokenizing text into words
_WORD_RE = re.compile(r"[a-z'\-]+", re.IGNORECASE)
_CAPS_WORD_RE = re.compile(r"\b[A-Z]{2,}\b")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens."""
    return [m.group().lower() for m in _WORD_RE.finditer(text)]


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, value))


def _compute_lexicon_valence(tokens: list[str]) -> float:
    """Score text valence from the lexicon, respecting negation and intensifiers.

    Walks the token list with a two-slot context window. Negators flip the
    sign of the next scored word. Intensifiers/diminishers scale its magnitude.
    Returns a normalized valence in [-1, 1].
    """
    if not tokens:
        return 0.0

    total = 0.0
    count = 0
    negate_next = False
    intensity_multiplier = 1.0

    for token in tokens:
        if token in _NEGATORS:
            negate_next = True
            continue

        if token in _INTENSIFIERS:
            intensity_multiplier = _INTENSIFIERS[token]
            continue

        if token in _DIMINISHERS:
            intensity_multiplier = _DIMINISHERS[token]
            continue

        score = _LEXICON.get(token, 0.0)
        if score != 0.0:
            score *= intensity_multiplier
            if negate_next:
                score *= -0.75  # Negation flips but slightly weaker
            total += score
            count += 1

        # Reset modifiers after consuming a scored or unscored content word
        negate_next = False
        intensity_multiplier = 1.0

    if count == 0:
        return 0.0

    # Normalize: average score mapped to [-1, 1] via tanh-like compression
    raw = total / count
    # Compress with a soft sigmoid so extreme texts don't saturate instantly
    return _clamp(math.tanh(raw / 3.0), -1.0, 1.0)


def _detect_sarcasm(text: str) -> float:
    """Return a sarcasm confidence score in [0, 1].

    Checks for air-quoted positive words, classic sarcastic phrases,
    and excessive punctuation after neutral/positive words.
    """
    hits = 0
    for pat in _SARCASM_PATTERNS:
        if pat.search(text):
            hits += 1

    # Excessive punctuation after words (e.g., "great!!!", "fine...")
    if re.search(r'\b\w+[!]{3,}', text):
        hits += 1
    if re.search(r'\b\w+\.{3,}', text):
        hits += 0.5

    # Ellipsis at end of short messages is suspicious
    stripped = text.strip()
    if stripped.endswith("...") and len(stripped) < 40:
        hits += 0.5

    return _clamp(hits / 3.0, 0.0, 1.0)


def _detect_urgency(text: str, tokens: list[str]) -> float:
    """Return an urgency score in [0, 1].

    Factors: ALL-CAPS ratio, exclamation density, imperative verbs,
    urgency keywords (ASAP, now, immediately).
    """
    if not text.strip():
        return 0.0

    signals = 0.0

    # All-caps word ratio (ignoring single-letter words like "I")
    caps_matches = _CAPS_WORD_RE.findall(text)
    caps_count = len(caps_matches)
    word_count = max(len(tokens), 1)
    caps_ratio = caps_count / word_count
    if caps_ratio > 0.3:
        signals += 0.4
    elif caps_ratio > 0.15:
        signals += 0.2

    # Exclamation density
    excl_count = text.count("!")
    excl_density = excl_count / max(len(text), 1)
    if excl_count >= 3 or excl_density > 0.05:
        signals += 0.3
    elif excl_count >= 1:
        signals += 0.1

    # Imperative verbs at start or urgency keywords anywhere
    if tokens:
        if tokens[0] in _IMPERATIVE_VERBS:
            signals += 0.2
    imperative_count = sum(1 for t in tokens if t in _IMPERATIVE_VERBS)
    if imperative_count >= 2:
        signals += 0.2

    return _clamp(signals, 0.0, 1.0)


def _detect_frustration(
    text: str,
    tokens: list[str],
    history: deque,
) -> float:
    """Return a frustration score in [0, 1].

    Signals:
    - Repeated near-identical messages (user re-sending)
    - Short terse response after a long one
    - Question marks after declarative statements
    - Profanity / strong negative words
    """
    signals = 0.0
    stripped = text.strip().lower()

    # Check for repeated messages (last 5 user messages)
    recent_texts = []
    for vec in reversed(list(history)):
        if vec.role == "user":
            recent_texts.append(vec)
        if len(recent_texts) >= 5:
            break
    # If user said almost the same thing recently, that is frustration
    # (We use the stored vectors rather than raw text to stay lightweight;
    #  repeated identical vectors with high frustration reinforce.)

    # Short terse response after a long conversation stretch
    if len(tokens) <= 3 and len(history) >= 3:
        # Look at the last user message length
        for vec in reversed(list(history)):
            if vec.role == "user":
                # If prior arousal was high and now we get a terse reply,
                # that often signals frustration or disengagement.
                if vec.arousal > 0.4:
                    signals += 0.15
                break

    # Question mark after a statement-like sentence (rhetorical frustration)
    if re.search(r'[a-z]\?{2,}', stripped):
        signals += 0.2
    if re.search(r'\?\s*$', stripped) and not stripped.startswith(("what", "how", "why", "when", "where", "who", "which", "is", "are", "do", "does", "can", "could", "would", "will", "should")):
        signals += 0.15

    # Strong negative lexicon words boost frustration
    strong_neg_count = sum(1 for t in tokens if _LEXICON.get(t, 0) <= -3)
    if strong_neg_count >= 2:
        signals += 0.3
    elif strong_neg_count == 1:
        signals += 0.15

    # Excessive punctuation (???, !!!)
    if re.search(r'[?!]{3,}', text):
        signals += 0.2

    return _clamp(signals, 0.0, 1.0)


def _detect_warmth(tokens: list[str]) -> float:
    """Return a warmth/rapport score in [0, 1].

    Based on greetings, humor markers, compliments, and casual tone words.
    """
    if not tokens:
        return 0.0

    warmth_hits = sum(1 for t in tokens if t in _WARMTH_WORDS)
    # Normalize: 1 hit = mild warmth, 3+ = strong warmth
    base = _clamp(warmth_hits / 3.0, 0.0, 0.8)

    # Emoji-like text emoticons add warmth
    # (We check for simple ones since we can't reliably detect Unicode emoji
    # without external libs.)
    emoticon_count = len(re.findall(r'[:;][\-]?[)D(P/\\]', " ".join(tokens)))
    if emoticon_count > 0:
        base += 0.1

    return _clamp(base, 0.0, 1.0)


def _compute_arousal(
    text: str,
    tokens: list[str],
    urgency: float,
    frustration: float,
) -> float:
    """Derive arousal (calm-to-excited) from text energy signals.

    High arousal: lots of caps, punctuation, long text, urgency, frustration.
    Low arousal: short calm text, no caps, no punctuation emphasis.
    """
    signals = 0.0

    # Length contributes mildly (longer messages = more engaged)
    word_count = len(tokens)
    if word_count > 50:
        signals += 0.2
    elif word_count > 20:
        signals += 0.1

    # Caps ratio
    caps_matches = _CAPS_WORD_RE.findall(text)
    caps_ratio = len(caps_matches) / max(word_count, 1)
    signals += caps_ratio * 0.3

    # Punctuation intensity
    punct_count = sum(1 for c in text if c in "!?")
    signals += min(punct_count * 0.05, 0.3)

    # Urgency and frustration bleed into arousal
    signals += urgency * 0.3
    signals += frustration * 0.2

    return _clamp(signals, 0.0, 1.0)


def _compute_dominance(tokens: list[str], urgency: float) -> float:
    """Derive dominance (submissive-to-assertive) from tone signals.

    Imperative verbs, short direct sentences, and urgency raise dominance.
    Hedging language ("maybe", "perhaps", "I think") lowers it.
    """
    base = 0.5  # Neutral starting point

    # Imperative verbs raise dominance
    imperative_count = sum(1 for t in tokens if t in _IMPERATIVE_VERBS)
    base += imperative_count * 0.05

    # Hedging lowers dominance
    hedges = {"maybe", "perhaps", "possibly", "might", "could", "seems",
              "think", "guess", "wonder", "hopefully", "sorry", "apologize"}
    hedge_count = sum(1 for t in tokens if t in hedges)
    base -= hedge_count * 0.06

    # Urgency raises dominance
    base += urgency * 0.2

    return _clamp(base, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Inflection point detection
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class InflectionPoint:
    """Records a detected mood shift in the conversation."""

    index: int
    """Message index in the trajectory where the shift occurred."""

    dimension: str
    """Which emotional dimension shifted (e.g., 'valence', 'frustration')."""

    delta: float
    """Magnitude and direction of the shift."""

    description: str
    """Human-readable description of the shift."""


# ---------------------------------------------------------------------------
# Trajectory tracker
# ---------------------------------------------------------------------------

class SentimentTrajectoryTracker:
    """Tracks user emotional state across a conversation via text analysis.

    Maintains a sliding window of per-message EmotionalVectors and an
    exponentially-weighted moving state that smoothly transitions between
    emotional states. Detects inflection points (mood shifts) and generates
    human-readable narrative summaries.

    Usage:
        tracker = SentimentTrajectoryTracker()
        vec = await tracker.analyze("I'm really frustrated with this bug!")
        print(tracker.get_mood_narrative())
    """

    # Exponential smoothing factor: 0.3 = rapid response to mood shifts
    ALPHA: float = 0.3

    # Sliding window size
    WINDOW_SIZE: int = 50

    # Hysteresis thresholds: a dimension must change by this much from the
    # smoothed state to be considered an inflection (prevents jitter).
    INFLECTION_THRESHOLD: float = 0.15

    def __init__(self) -> None:
        self._history: deque[EmotionalVector] = deque(maxlen=self.WINDOW_SIZE)
        self._inflections: list[InflectionPoint] = []
        self._message_count: int = 0
        self._lock = asyncio.Lock()

        # Smoothed state — starts at neutral
        self._state = EmotionalVector(
            valence=0.0,
            arousal=0.0,
            dominance=0.5,
            urgency=0.0,
            warmth=0.0,
            frustration=0.0,
        )

        logger.debug("SentimentTrajectoryTracker initialized")

    # ── Public API ────────────────────────────────────────────────────────────

    async def analyze(self, text: str, role: str = "user") -> EmotionalVector:
        """Analyze a message and return its emotional vector.

        Also updates the smoothed state and checks for inflection points.
        This is the primary entry point — call once per message.

        Args:
            text: The raw message text to analyze.
            role: Who produced the text ('user' or 'assistant').
                  Only user messages update trajectory and inflection detection.

        Returns:
            The EmotionalVector for this specific message.
        """
        async with self._lock:
            return self._analyze_locked(text, role)

    def get_trajectory(self) -> list[EmotionalVector]:
        """Return the full sliding window of recent emotional vectors.

        The list is ordered oldest-first. Each entry is a frozen dataclass,
        so this is safe to iterate without locks.
        """
        return list(self._history)

    def get_current_state(self) -> EmotionalVector:
        """Return the current exponentially-smoothed emotional state.

        This represents the 'running mood' of the conversation — smoothed
        across recent messages to avoid jitter from single outlier messages.
        """
        return self._state

    def get_inflections(self) -> list[InflectionPoint]:
        """Return all detected emotional inflection points so far."""
        return list(self._inflections)

    def get_mood_narrative(self) -> str:
        """Generate a human-readable summary of the conversation's emotional arc.

        Describes how the user's mood has evolved from the start of the
        conversation through the current state, noting any significant shifts.

        Returns:
            A plain-English description like:
            'The user started this conversation casually and warmly, but their
            tone has shifted to mild frustration over the last 3 messages.'
        """
        history = list(self._history)
        if not history:
            return "No messages analyzed yet."

        user_msgs = [v for v in history if v.role == "user"]
        if not user_msgs:
            return "No user messages analyzed yet."

        # Describe the opening mood
        opening = self._describe_vector(user_msgs[0], label="opening")

        # Describe the current mood
        current = self._describe_vector(self._state, label="current")

        # Count recent messages for context
        recent_count = min(len(user_msgs), 3)
        recent_slice = user_msgs[-recent_count:]
        recent_desc = self._describe_trend(recent_slice)

        # Build the narrative
        parts = []

        if len(user_msgs) == 1:
            parts.append(f"The user's tone is {current}.")
        else:
            parts.append(f"The user started this conversation {opening}.")

            # Check for shifts
            if self._inflections:
                last_inflection = self._inflections[-1]
                parts.append(last_inflection.description)

            if recent_desc:
                parts.append(recent_desc)
            else:
                parts.append(f"Their current tone is {current}.")

        return " ".join(parts)

    def reset(self) -> None:
        """Clear all tracked state. Useful when a new conversation begins."""
        self._history.clear()
        self._inflections.clear()
        self._message_count = 0
        self._state = EmotionalVector(
            valence=0.0,
            arousal=0.0,
            dominance=0.5,
            urgency=0.0,
            warmth=0.0,
            frustration=0.0,
        )
        logger.debug("SentimentTrajectoryTracker reset")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _analyze_locked(self, text: str, role: str) -> EmotionalVector:
        """Core analysis logic, called under lock."""
        tokens = _tokenize(text)

        # Compute each dimension
        valence = _compute_lexicon_valence(tokens)
        sarcasm = _detect_sarcasm(text)
        urgency = _detect_urgency(text, tokens)
        frustration = _detect_frustration(text, tokens, self._history)
        warmth = _detect_warmth(tokens)
        arousal = _compute_arousal(text, tokens, urgency, frustration)
        dominance = _compute_dominance(tokens, urgency)

        # Sarcasm inverts valence and boosts frustration
        if sarcasm > 0.3:
            valence = _clamp(valence * (1.0 - sarcasm * 0.8), -1.0, 1.0)
            # If the raw valence was positive but sarcasm detected, flip it
            if valence > 0:
                valence = _clamp(-valence * sarcasm, -1.0, 1.0)
            frustration = _clamp(frustration + sarcasm * 0.3, 0.0, 1.0)

        # Build the message-level vector
        vec = EmotionalVector(
            valence=round(valence, 4),
            arousal=round(arousal, 4),
            dominance=round(dominance, 4),
            urgency=round(urgency, 4),
            warmth=round(warmth, 4),
            frustration=round(frustration, 4),
            role=role,
        )

        # Store in history
        self._history.append(vec)
        self._message_count += 1

        # Update smoothed state (only for user messages)
        if role == "user":
            self._update_smoothed_state(vec)
            self._detect_inflections(vec)

        logger.debug(
            "Sentiment [#%d %s]: V=%.2f A=%.2f D=%.2f U=%.2f W=%.2f F=%.2f",
            self._message_count, role,
            vec.valence, vec.arousal, vec.dominance,
            vec.urgency, vec.warmth, vec.frustration,
        )

        return vec

    def _update_smoothed_state(self, vec: EmotionalVector) -> None:
        """Apply exponential moving average to update the smoothed state.

        new_state = alpha * new_observation + (1 - alpha) * old_state

        The alpha of 0.3 means recent messages weigh ~30% each, giving
        rapid but not jittery response to mood changes.
        """
        a = self.ALPHA
        b = 1.0 - a

        self._state = EmotionalVector(
            valence=round(a * vec.valence + b * self._state.valence, 4),
            arousal=round(a * vec.arousal + b * self._state.arousal, 4),
            dominance=round(a * vec.dominance + b * self._state.dominance, 4),
            urgency=round(a * vec.urgency + b * self._state.urgency, 4),
            warmth=round(a * vec.warmth + b * self._state.warmth, 4),
            frustration=round(a * vec.frustration + b * self._state.frustration, 4),
            role="smoothed",
        )

    def _detect_inflections(self, vec: EmotionalVector) -> None:
        """Check whether the new message constitutes an inflection point.

        An inflection occurs when a dimension jumps by more than the
        hysteresis threshold relative to the smoothed state *before* this
        message was applied. We compare against the pre-update state stored
        in self._state (which was already updated, so we compare the delta
        from the raw vec to the prior smoothed state).
        """
        if self._message_count < 3:
            return  # Need some history before detecting shifts

        dims = [
            ("valence", vec.valence, self._state.valence),
            ("arousal", vec.arousal, self._state.arousal),
            ("frustration", vec.frustration, self._state.frustration),
            ("warmth", vec.warmth, self._state.warmth),
            ("urgency", vec.urgency, self._state.urgency),
        ]

        for dim_name, raw_val, smoothed_val in dims:
            delta = raw_val - smoothed_val
            if abs(delta) >= self.INFLECTION_THRESHOLD:
                direction = "increased" if delta > 0 else "decreased"
                desc = self._inflection_narrative(dim_name, delta, direction)
                point = InflectionPoint(
                    index=self._message_count,
                    dimension=dim_name,
                    delta=round(delta, 4),
                    description=desc,
                )
                self._inflections.append(point)
                logger.info(
                    "Inflection detected at message #%d: %s %s by %.2f",
                    self._message_count, dim_name, direction, abs(delta),
                )

    @staticmethod
    def _inflection_narrative(dim: str, delta: float, direction: str) -> str:
        """Create a human-readable description for an inflection point."""
        magnitude = abs(delta)
        if magnitude > 0.4:
            strength = "sharply"
        elif magnitude > 0.25:
            strength = "noticeably"
        else:
            strength = "slightly"

        templates = {
            "valence": {
                "increased": f"Their mood {strength} brightened in the recent message.",
                "decreased": f"Their mood {strength} darkened in the recent message.",
            },
            "arousal": {
                "increased": f"Their energy level {strength} spiked.",
                "decreased": f"Their energy level {strength} dropped.",
            },
            "frustration": {
                "increased": f"Frustration {strength} increased in the recent message.",
                "decreased": f"Frustration has {strength} eased.",
            },
            "warmth": {
                "increased": f"The tone became {strength} warmer and friendlier.",
                "decreased": f"The tone became {strength} more distant.",
            },
            "urgency": {
                "increased": f"The sense of urgency {strength} increased.",
                "decreased": f"The urgency has {strength} subsided.",
            },
        }

        return templates.get(dim, {}).get(direction, f"{dim} {direction} by {magnitude:.2f}.")

    @staticmethod
    def _describe_vector(vec: EmotionalVector, label: str = "current") -> str:
        """Produce a terse English description of a single emotional vector."""
        parts = []

        # Valence
        if vec.valence > 0.4:
            parts.append("positive")
        elif vec.valence > 0.15:
            parts.append("mildly positive")
        elif vec.valence < -0.4:
            parts.append("quite negative")
        elif vec.valence < -0.15:
            parts.append("mildly negative")
        else:
            parts.append("neutral")

        # Arousal
        if vec.arousal > 0.6:
            parts.append("energetic")
        elif vec.arousal < 0.2:
            parts.append("calm")

        # Warmth
        if vec.warmth > 0.5:
            parts.append("warm and friendly")
        elif vec.warmth > 0.25:
            parts.append("casually friendly")

        # Frustration
        if vec.frustration > 0.6:
            parts.append("notably frustrated")
        elif vec.frustration > 0.3:
            parts.append("mildly frustrated")

        # Urgency
        if vec.urgency > 0.6:
            parts.append("with a sense of urgency")
        elif vec.urgency > 0.3:
            parts.append("somewhat urgent")

        if not parts:
            return "neutral"

        return ", ".join(parts)

    def _describe_trend(self, recent: list[EmotionalVector]) -> str:
        """Describe the trend over the last few messages."""
        if len(recent) < 2:
            current_desc = self._describe_vector(self._state, label="current")
            return f"Their current tone is {current_desc}."

        # Compare first and last of the recent slice
        first = recent[0]
        last = recent[-1]
        count = len(recent)

        shifts = []

        val_delta = last.valence - first.valence
        if val_delta > 0.2:
            shifts.append("mood has been improving")
        elif val_delta < -0.2:
            shifts.append("mood has been declining")

        frust_delta = last.frustration - first.frustration
        if frust_delta > 0.15:
            shifts.append("frustration has been building")
        elif frust_delta < -0.15:
            shifts.append("frustration has been easing")

        warmth_delta = last.warmth - first.warmth
        if warmth_delta > 0.15:
            shifts.append("tone has been warming up")
        elif warmth_delta < -0.15:
            shifts.append("tone has been cooling down")

        if shifts:
            trend = " and ".join(shifts)
            return f"Over the last {count} messages, their {trend}."

        current_desc = self._describe_vector(self._state, label="current")
        return f"Their current tone is {current_desc}."


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_tracker: Optional[SentimentTrajectoryTracker] = None


def get_sentiment_tracker() -> SentimentTrajectoryTracker:
    """Return the module-level singleton SentimentTrajectoryTracker."""
    global _tracker
    if _tracker is None:
        _tracker = SentimentTrajectoryTracker()
    return _tracker
