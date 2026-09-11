"""core/voice/duplex/fillers.py — "uh…", "let me check that."

Aura's mind is a governed 32B pipeline. Some turns genuinely take seconds —
retrieval, tool use, deliberation. Dead air across those seconds reads as a
crash, and the user starts repeating themselves.

The important design constraint: a filler must never be theatre. Everything
here is keyed to *what she is actually doing right now*, taken from the
cognitive engine's own activity telemetry. When she says "let me look that
up", a web search is genuinely in flight. When she says "hm, let me think",
the deliberation lane is genuinely running.

That also makes fillers safe in a way sentences are not: they assert
nothing about the answer, so they cannot turn out to be false. The worst
case is that she sounds thoughtful about something that finished quickly.

Three escalating tiers, because a single filler repeated is worse than
silence:

    ~380 ms   a breath — "uh…", "mm…"        (covers ordinary think time)
    ~1.9 s    a reason — "let me check that"  (says what is taking the time)
    ~6.5 s    a status — "still with you"     (proves the line is alive)
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from enum import Enum
from core.runtime.the_laboratory import seeded

logger = logging.getLogger("Aura.Voice.Filler")


class ThinkingCause(Enum):
    """Why this turn is slow. Drives which words she uses."""

    UNKNOWN = "unknown"
    DELIBERATION = "deliberation"   # reasoning lane is working
    RETRIEVAL = "retrieval"         # memory / recall
    WEB_SEARCH = "web_search"
    TOOL_USE = "tool_use"           # terminal, files, OS
    IMAGE = "image"
    SELF_WORK = "self_work"         # introspection, self-modification
    UNCERTAINTY = "uncertainty"     # low confidence, weighing


# The engine's own activity keys (core/cognitive/state_machine.ACTIVITY_MAP)
# mapped to causes. This is the seam that keeps fillers honest.
ACTIVITY_TO_CAUSE: dict[str, ThinkingCause] = {
    "sovereign_browser": ThinkingCause.WEB_SEARCH,
    "sovereign_terminal": ThinkingCause.TOOL_USE,
    "sovereign_network": ThinkingCause.TOOL_USE,
    "file_operation": ThinkingCause.TOOL_USE,
    "os_manipulation": ThinkingCause.TOOL_USE,
    "manifest_to_device": ThinkingCause.TOOL_USE,
    "generate_image": ThinkingCause.IMAGE,
    "sovereign_imagination": ThinkingCause.IMAGE,
    "social_lurker": ThinkingCause.WEB_SEARCH,
    "self_improvement": ThinkingCause.SELF_WORK,
    "self_evolution": ThinkingCause.SELF_WORK,
    "memory_recall": ThinkingCause.RETRIEVAL,
    "deliberation": ThinkingCause.DELIBERATION,
}

# Tier 1 — a breath. Short, low-content, buys ~1 s.
_TIER1: tuple[str, ...] = ("uh…", "mm…", "hm…", "so…", "okay…")

# Tier 2 — names the real cause.
_TIER2: dict[ThinkingCause, tuple[str, ...]] = {
    ThinkingCause.UNKNOWN: ("let me think about that.", "hm, one sec.", "give me a moment."),
    ThinkingCause.DELIBERATION: (
        "let me think about that properly.",
        "hm — I want to get this right.",
        "okay, thinking it through.",
    ),
    ThinkingCause.RETRIEVAL: (
        "let me remember…",
        "hang on, I'm pulling that up.",
        "checking what I have on that.",
    ),
    ThinkingCause.WEB_SEARCH: (
        "let me look that up.",
        "I'm searching for that now.",
        "one sec, checking.",
    ),
    ThinkingCause.TOOL_USE: (
        "running that now.",
        "one moment, I'm on it.",
        "doing that — hang on.",
    ),
    ThinkingCause.IMAGE: ("I'm making that now.", "give me a moment, it's rendering."),
    ThinkingCause.SELF_WORK: ("hang on, I'm working on myself here.", "one sec, this one's internal."),
    ThinkingCause.UNCERTAINTY: (
        "hm, I'm not sure yet.",
        "let me weigh that.",
        "I want to be careful here.",
    ),
}

# Causes that are known to take seconds. Naming one of these immediately is
# more informative than a generic hesitation sound, and it is honest: the
# work really is in flight when she says so.
_ANNOUNCE_IMMEDIATELY = frozenset({
    ThinkingCause.WEB_SEARCH,
    ThinkingCause.TOOL_USE,
    ThinkingCause.IMAGE,
    ThinkingCause.RETRIEVAL,
    ThinkingCause.SELF_WORK,
})

# Tier 3 — proves the line is alive on genuinely long turns.
_TIER3: tuple[str, ...] = (
    "still with you — this one's taking a bit.",
    "sorry, still working on it.",
    "bear with me, nearly there.",
)


@dataclass(slots=True)
class FillerUtterance:
    text: str
    tier: int
    cause: ThinkingCause
    # Fillers sit slightly under her normal speech: they are asides, not
    # statements, and full-volume "uh" sounds like a glitch.
    gain: float = 0.72


class FillerReflex:
    """Emits thinking sounds on a schedule, keyed to the real cause.

    Stateful across a single turn: tracks which tiers have fired so the same
    words never repeat inside one wait.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or seeded("voice.fillers")
        self._fired: set[int] = set()
        self._cause = ThinkingCause.UNKNOWN
        self._recent: list[str] = []
        self._announce_now = False

    def begin_turn(self) -> None:
        self._fired.clear()
        self._cause = ThinkingCause.UNKNOWN
        self._announce_now = False

    def observe_activity(self, activity_key: str) -> None:
        """Update the cause from live engine telemetry.

        Called whenever the cognitive engine reports what it is doing, so a
        tier-2 filler describes the work actually in flight at that instant.
        """
        cause = ACTIVITY_TO_CAUSE.get((activity_key or "").strip().lower())
        if cause is not None:
            self._cause = cause
            # Knowing *why* she is slow is better information than knowing
            # *that* she is slow, so a long-running cause should announce
            # itself as soon as it is known rather than waiting out a fixed
            # timer. "Let me look that up" at 300 ms beats "uh…" at 380 ms
            # followed by the same sentence 1.5 s later.
            if cause in _ANNOUNCE_IMMEDIATELY:
                self._announce_now = True

    def set_cause(self, cause: ThinkingCause) -> None:
        self._cause = cause

    @property
    def cause(self) -> ThinkingCause:
        return self._cause

    def due(self, elapsed_ms: float, *, first: float, second: float, third: float) -> FillerUtterance | None:
        """Return a filler if this wait has crossed an unfired tier."""
        tier: int
        # A known long-running cause skips straight to naming itself, and
        # marks tier 1 spent so she does not say "uh…" after already having
        # explained what she is doing.
        if self._announce_now and 2 not in self._fired:
            self._announce_now = False
            self._fired.add(1)
            self._fired.add(2)
            return FillerUtterance(
                text=self._pick(_TIER2.get(self._cause) or _TIER2[ThinkingCause.UNKNOWN]),
                tier=2,
                cause=self._cause,
                gain=0.78,
            )

        if elapsed_ms >= third and 3 not in self._fired:
            tier = 3
        elif elapsed_ms >= second and 2 not in self._fired:
            tier = 2
        elif elapsed_ms >= first and 1 not in self._fired:
            tier = 1
        else:
            return None

        self._fired.add(tier)
        if tier == 1:
            text = self._pick(_TIER1)
            gain = 0.62  # a breath, not a statement
        elif tier == 2:
            text = self._pick(_TIER2.get(self._cause) or _TIER2[ThinkingCause.UNKNOWN])
            gain = 0.78
        else:
            text = self._pick(_TIER3)
            gain = 0.82

        return FillerUtterance(text=text, tier=tier, cause=self._cause, gain=gain)

    def _pick(self, options: tuple[str, ...]) -> str:
        fresh = [o for o in options if o not in self._recent[-3:]]
        choice = self._rng.choice(fresh or list(options))
        self._recent.append(choice)
        if len(self._recent) > 6:
            self._recent.pop(0)
        return choice
