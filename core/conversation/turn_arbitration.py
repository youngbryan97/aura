"""One place that knows what happened in a turn.

Every mechanism in the response path is locally correct and they interfere
anyway, because nothing observes the interference. The failures share a shape:

* an honesty gate suppresses a good answer, and the reason is in a log line
  nobody correlates with the apology the user got;
* two lanes both produce a reply and whichever finishes last wins, so quality
  depends on timing rather than merit;
* a deep lane returns after a quick lane already answered, and speaks into a
  conversation that has moved on;
* a gate's decision is invisible to the next gate, so two of them can undo each
  other's work within one turn.

None of these is fixable inside the mechanism that caused it, because each one
is behaving correctly on the information it has. What is missing is a record of
the turn as a whole: which lanes ran, which gates fired, what each suppressed,
and which answer was actually served.

With that record, three things become decidable instead of accidental. A late
lane can be refused by rule rather than by luck. Competing candidates can be
ranked by declared precedence rather than arrival order. And suppression stops
being anonymous — the gate that discarded an answer is named, alongside the
answer it discarded, so "she said she couldn't answer" can be traced to the
thing that decided she couldn't.

Deliberately a ledger and an arbiter, not a controller. It does not run lanes
or call gates; it records what they did and answers questions about the turn.
Anything that wants to change behaviour on the strength of it asks — which is
why `should_serve` returns a reason rather than raising.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum

_RECOVERABLE = (RuntimeError, AttributeError, TypeError, ValueError, KeyError, ImportError)


def _record_custody_degradation(exc: BaseException) -> None:
    """Report a custody-check failure without importing errors at module load."""

    try:
        from core.runtime.errors import record_degradation

        record_degradation(
            "turn_arbitration.fact_custody",
            exc,
            severity="warning",
            action="a text mutation went unchecked against the turn's held facts",
        )
    # not a failure: the comment beside it says it: the reporter itself is missing.
    except _RECOVERABLE:  # pragma: no cover - the reporter itself is missing
        pass


class LanePrecedence(IntEnum):
    """Which answer wins when more than one is available.

    Ordered by how much of the mind produced it, not by how fast it arrived.
    A quick reply that beat a full one to the finish line is still the lesser
    answer; before this, it won simply for being first.
    """

    FALLBACK = 10          # canned or templated text
    REPAIR = 20            # bounded repair machinery authored it
    QUICK = 30             # the fast conversational lane
    FULL = 40              # the full cognitive path
    DEEP = 50              # deliberate/deep reasoning
    TOOL_VERIFIED = 60     # backed by an executed, verified action


@dataclass(frozen=True)
class Suppression:
    """A gate that removed or replaced text, and what it removed."""

    gate: str
    reason: str
    suppressed_chars: int
    replaced_with_chars: int
    at: float = field(default_factory=lambda: time.time())
    #: What was removed, bounded. Sizes let this be counted; the text lets it
    #: be undone. A refusal site holding "the engine produced no acceptable
    #: reply" is often standing next to the acceptable reply a gate took, and
    #: without the text there is nothing to hand back.
    suppressed_text: str = ""

    def describe(self) -> str:
        return (
            f"{self.gate} suppressed {self.suppressed_chars} chars "
            f"({self.reason}) and served {self.replaced_with_chars}"
        )


@dataclass(frozen=True)
class Candidate:
    """One answer a lane produced."""

    lane: str
    precedence: LanePrecedence
    chars: int
    authentic: bool
    at: float = field(default_factory=lambda: time.time())


@dataclass
class TurnLedger:
    """What happened during one turn, from first lane to served reply."""

    turn_id: str
    started_at: float = field(default_factory=lambda: time.time())
    candidates: list[Candidate] = field(default_factory=list)
    suppressions: list[Suppression] = field(default_factory=list)
    served_lane: str = ""
    served_at: float = 0.0

    @property
    def answered(self) -> bool:
        return bool(self.served_lane)

    def record_candidate(
        self, lane: str, precedence: LanePrecedence, *, chars: int, authentic: bool = True
    ) -> None:
        self.candidates.append(
            Candidate(lane=lane, precedence=precedence, chars=chars, authentic=authentic)
        )

    def record_suppression(
        self, gate: str, reason: str, *, before: str, after: str
    ) -> None:
        """A gate changed the outgoing text. Both sizes matter.

        An apology that replaces 900 characters of real answer and one that
        tidies 3 characters of whitespace are not the same event, and only the
        sizes distinguish them after the fact.
        """
        text = str(before or "")
        self.suppressions.append(
            Suppression(
                gate=str(gate),
                reason=str(reason)[:200],
                suppressed_chars=len(text),
                replaced_with_chars=len(str(after or "")),
                suppressed_text=text[:_MAX_SUPPRESSED_TEXT],
            )
        )
        # A turn that trips fifty gates is a turn in trouble, not a turn worth
        # remembering fifty times over. Keep the most recent.
        if len(self.suppressions) > _MAX_SUPPRESSIONS:
            del self.suppressions[:-_MAX_SUPPRESSIONS]
        # Sizes say a gate took 900 characters. They cannot say it took the
        # ANSWER. Every stage that rewrites outgoing text arrives here, which
        # makes this the one place a held fact can be checked across all of
        # them at once — and the only place where the stage responsible is
        # still known. A new gate written next year is covered on the day it
        # is written rather than the day someone remembers to instrument it.
        try:
            from core.runtime.fact_custody import inspect_mutation

            inspect_mutation(str(gate), text, str(after or ""))
        except _RECOVERABLE as exc:
            # Custody is an observer of this seam, never a gate on it. A ledger
            # that can refuse a turn is a ledger that can lose one.
            _record_custody_degradation(exc)

    def record_served(self, lane: str) -> None:
        self.served_lane = str(lane)
        self.served_at = time.time()

    def should_serve(self, lane: str, precedence: LanePrecedence) -> tuple[bool, str]:
        """May this lane speak, given what has already happened this turn?

        The rule is not "first wins" and not "best wins", but "the turn is
        answered once". A lane that arrives after the turn was served is late,
        and a late answer is not a better answer — it is a second answer to a
        conversation that has moved on. It is refused with a reason, so the
        caller can log it rather than discovering a duplicate reply in the UI.
        """
        if not self.answered:
            return True, "the turn is unanswered"
        if lane == self.served_lane:
            return False, "this lane already answered this turn"
        served = next(
            (item for item in self.candidates if item.lane == self.served_lane), None
        )
        if served is not None and precedence > served.precedence:
            # Better answers do not get to arrive late either. Serving it now
            # would put two replies in front of the person, and the second one
            # answering a question they have already moved past.
            return False, (
                f"{lane} outranks the served {self.served_lane} but arrived "
                f"{time.time() - self.served_at:.1f}s after the turn was answered"
            )
        return False, f"the turn was already answered by {self.served_lane}"

    def best_candidate(self) -> Candidate | None:
        """The answer that should win, by merit rather than by timing."""
        usable = [item for item in self.candidates if item.chars > 0 and item.authentic]
        if not usable:
            usable = [item for item in self.candidates if item.chars > 0]
        if not usable:
            return None
        return max(usable, key=lambda item: (int(item.precedence), item.chars))

    def narrative(self) -> str:
        """The turn in one line, for a log or a health report."""
        parts = [f"turn={self.turn_id}"]
        if self.candidates:
            parts.append(
                "candidates=" + ",".join(
                    f"{item.lane}({item.precedence.name}/{item.chars}c)"
                    for item in self.candidates
                )
            )
        if self.suppressions:
            parts.append(
                "suppressed=" + "; ".join(item.describe() for item in self.suppressions)
            )
        parts.append(f"served={self.served_lane or 'nothing'}")
        return " | ".join(parts)

    def recoverable_text(self) -> str:
        """The largest thing a gate took that is still worth saying, or "".

        No judgement of quality here — the caller reassesses before serving.
        This only answers "did this turn produce something substantial that
        never reached the person", which is the question a refusal site cannot
        otherwise ask.
        """
        losses = [item for item in self.lost_work() if item.suppressed_text.strip()]
        if not losses:
            return ""
        return max(losses, key=lambda item: item.suppressed_chars).suppressed_text

    def lost_work(self) -> list[Suppression]:
        """Suppressions that discarded more than they served.

        The recurring, expensive failure in this codebase is a good answer
        produced, discarded by a gate, and reported to the user as an
        infrastructure problem. This names those events without judging them —
        some are correct — so they can be counted rather than rediscovered.
        """
        return [
            item
            for item in self.suppressions
            if item.suppressed_chars > max(40, item.replaced_with_chars * 2)
        ]


#: Enough for a long reply; small enough that 64 turns of them stay trivial.
_MAX_SUPPRESSED_TEXT = 8000
_MAX_SUPPRESSIONS = 24

_LOCK = threading.Lock()
_LEDGERS: dict[str, TurnLedger] = {}
_MAX_LEDGERS = 64


def ledger_for(turn_id: str) -> TurnLedger:
    """The ledger for this turn, created on first use.

    Bounded: a turn ledger is for the turn and its immediate aftermath, not a
    history. Keeping every one would make this the memory leak it exists to
    help diagnose.
    """
    key = str(turn_id or "").strip()
    if not key:
        raise ValueError("turn arbitration requires an exact turn identity")
    with _LOCK:
        ledger = _LEDGERS.get(key)
        if ledger is None:
            ledger = TurnLedger(turn_id=key)
            _LEDGERS[key] = ledger
            if len(_LEDGERS) > _MAX_LEDGERS:
                for stale in sorted(_LEDGERS, key=lambda k: _LEDGERS[k].started_at)[
                    : len(_LEDGERS) - _MAX_LEDGERS
                ]:
                    _LEDGERS.pop(stale, None)
        return ledger


def forget_turn(turn_id: str) -> None:
    with _LOCK:
        _LEDGERS.pop(str(turn_id or ""), None)


def reset_turn_ledgers_for_test() -> None:
    with _LOCK:
        _LEDGERS.clear()


__all__ = [
    "Candidate",
    "LanePrecedence",
    "Suppression",
    "TurnLedger",
    "forget_turn",
    "ledger_for",
    "reset_turn_ledgers_for_test",
]
