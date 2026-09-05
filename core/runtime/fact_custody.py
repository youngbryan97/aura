"""Custody of an established fact across the stages that rewrite the reply.

The defect this exists for has one shape and many instances:

    Stored ⇏ Indexed ⇏ Retrieved ⇏ Selected ⇏ Rendered ⇏ Delivered

A directory read produced a verified count of 9. The receipts survived a
restart. Retrieval found them. A correction stated the number. And the person
was told seventeen — because a later shaping pass read the correction as
off-topic and stripped it, and a later repair pass replaced the whole reply
with a sentence saying the count had never been taken. Every one of those
stages was locally correct. Nothing was watching the composition.

The general statement of the failure is that a fact one part of the system
knows does not arrive intact at the part that speaks. It is not a knowledge
failure, a retrieval failure, or a reasoning failure, and no amount of work
inside any single stage fixes it, because no single stage is wrong.

So this module makes custody an invariant rather than a hope. A stage that
establishes a fact from evidence puts it in the turn's custody set. Every
subsequent text mutation — and they all pass through one seam — is checked
against that set. A mutation that removes a held fact from the text, or that
states a competing value for it, is a **custody break**, attributed to the
named stage that did it. At the terminal boundary a live break on a fact with
real evidence behind it is repaired from the held value.

Three properties make this different from another honesty guard:

**Facts come from producers, not from prose.** A stage registers what it
established, with the evidence ids and the value it measured. Nothing here
parses free text to discover what Aura believes; it only checks whether an
already-known value survived. That is a bounded string question, not an open
natural-language one, and it is the reason this generalises to any fact any
stage cares to hold rather than to a list of phrasings.

**Detection is sentence-scoped and value-typed.** A fact is present when a
sentence mentions its subject AND states its value. It is contradicted when a
sentence mentions its subject and states a *different* value of the same kind.
A number elsewhere in the reply about something else is neither.

**Attribution is by construction.** The break names the stage between the two
checkpoints, because the check runs on that stage's own before/after pair. The
output is "chat.response_repair dropped dir_py_count=9", not "the reply is
wrong" — which is the difference between an instrument and a complaint.

What this does NOT do: it does not decide what Aura should say, rank
candidates, or judge whether a fact is worth stating. A fact nobody held is a
fact nobody checks, and holding one is a deliberate act by the stage that has
the evidence.
"""

from __future__ import annotations

import enum
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from core.runtime.lockdep import LockRank, checked_lock
from core.runtime.turn_outcome import VerificationGrade, current_turn

__all__ = [
    "BreakKind",
    "CustodyBreak",
    "CustodySet",
    "HeldFact",
    "Restoration",
    "custody_for",
    "custody_report",
    "current_custody",
    "forget_custody",
    "hold_fact",
    "inspect_mutation",
    "reset_custody_for_test",
    "restore_held_facts",
]

logger = logging.getLogger(__name__)

#: Facts per turn. A turn holding more than this is not being served by a
#: custody problem, and an unbounded set on a hot path is its own defect.
_MAX_FACTS = 64
#: Breaks kept per turn. Same reasoning; the first ones are the informative
#: ones because later stages inherit the damage.
_MAX_BREAKS = 128
#: Turn custody sets retained. Matches the turn-ledger bound next door: enough
#: to answer "what happened on that turn" for the recent past, not an archive.
_MAX_TURNS = 64

#: Written-out integers, because the live failure said "seventeen" rather than
#: "17". Lexical normalisation of a value domain, bounded and inspectable —
#: not a phrasing list, which is the thing this module exists to stop needing.
_NUMBER_WORDS: dict[str, int] = {
    "zero": 0, "none": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000,
}

_WORD = re.compile(r"[A-Za-z0-9_.~/*-]+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_INTEGER = re.compile(r"\b\d[\d,]*\b")


class ValueKind(str, enum.Enum):
    """What kind of value this fact holds, which decides how it is compared.

    A count and a filesystem path are contradicted by different things. TEXT
    facts are contradicted only by absence, because "a different string in a
    sentence about the same subject" is not evidence of disagreement the way a
    different integer is.
    """

    NUMBER = "number"
    TEXT = "text"


class BreakKind(str, enum.Enum):
    DROPPED = "dropped"
    CONTRADICTED = "contradicted"


def _tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in _WORD.finditer(text or "")}


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT.split(text or "") if part.strip()]


def _integers_in(text: str) -> set[int]:
    """Every integer a sentence states, in digits or in words.

    Word numbers are read singly rather than composed: "twenty two" yields
    {20, 2}, not {22}. Deliberate — an over-broad reading here produces false
    contradictions, and the compound case has never been the one that matters
    for a measured count.
    """

    found: set[int] = set()
    for match in _INTEGER.finditer(text or ""):
        try:
            found.add(int(match.group(0).replace(",", "")))
        except ValueError:
            continue
    for token in _tokens(text):
        if token in _NUMBER_WORDS:
            found.add(_NUMBER_WORDS[token])
    return found


@dataclass(frozen=True, slots=True)
class HeldFact:
    """One thing a stage established, and what counts as saying it.

    ``canonical_rendering`` is the sentence the runtime will restore if the
    fact is lost. The producer writes it because the producer is what has the
    evidence; a renderer that composed it here would be inventing the claim it
    is meant to be protecting.
    """

    subject: str
    predicate: str
    value: str
    subject_cues: tuple[str, ...]
    canonical_rendering: str
    established_by: str
    grade: VerificationGrade = VerificationGrade.ASSERTED
    kind: ValueKind = ValueKind.TEXT
    value_forms: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    established_at: float = field(default_factory=time.time)

    @property
    def key(self) -> tuple[str, str]:
        return (self.subject.strip().lower(), self.predicate.strip().lower())

    @property
    def restorable(self) -> bool:
        """Whether a break in this fact may rewrite outgoing text.

        OBSERVED is the bar because it is the grade that means something
        looked. A component asserting its own success is not entitled to
        overwrite what a later stage decided to say.
        """

        return self.grade >= VerificationGrade.OBSERVED

    def _numeric_value(self) -> int | None:
        if self.kind is not ValueKind.NUMBER:
            return None
        for candidate in (self.value, *self.value_forms):
            for number in _integers_in(str(candidate)):
                return number
        return None

    def mentions_subject(self, sentence: str) -> bool:
        tokens = _tokens(sentence)
        return any(cue.strip().lower() in tokens for cue in self.subject_cues if cue.strip())

    def states_value(self, sentence: str) -> bool:
        if self.kind is ValueKind.NUMBER:
            mine = self._numeric_value()
            return mine is not None and mine in _integers_in(sentence)
        haystack = sentence.lower()
        return any(
            str(form).strip().lower() in haystack
            for form in (self.value, *self.value_forms)
            if str(form).strip()
        )

    def competing_values(self, sentence: str) -> tuple[str, ...]:
        """Values of this fact's own kind that the sentence states instead."""

        if self.kind is not ValueKind.NUMBER:
            return ()
        mine = self._numeric_value()
        if mine is None:
            return ()
        return tuple(
            str(number) for number in sorted(_integers_in(sentence)) if number != mine
        )

    def present_in(self, text: str) -> bool:
        return any(
            self.mentions_subject(sentence) and self.states_value(sentence)
            for sentence in _sentences(text)
        )

    def contradicting_sentence(self, text: str) -> str:
        """The first sentence that talks about this subject and gets it wrong."""

        for sentence in _sentences(text):
            if not self.mentions_subject(sentence):
                continue
            if self.states_value(sentence):
                continue
            if self.competing_values(sentence):
                return sentence
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "value": self.value,
            "kind": self.kind.value,
            "grade": self.grade.value,
            "restorable": self.restorable,
            "established_by": self.established_by,
            "evidence": list(self.evidence[:8]),
            "established_at": self.established_at,
        }


@dataclass(frozen=True, slots=True)
class CustodyBreak:
    """A named stage lost or contradicted a held fact."""

    fact: HeldFact
    stage: str
    kind: BreakKind
    detail: str
    at: float = field(default_factory=time.time)

    def describe(self) -> str:
        return (
            f"{self.stage} {self.kind.value} "
            f"{self.fact.subject}.{self.fact.predicate}={self.fact.value}"
            + (f" ({self.detail})" if self.detail else "")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "kind": self.kind.value,
            "detail": self.detail[:240],
            "at": self.at,
            "fact": self.fact.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class Restoration:
    """The result of enforcing custody on the text about to be spoken."""

    text: str
    changed: bool
    restored: tuple[CustodyBreak, ...] = ()
    unrestored: tuple[CustodyBreak, ...] = ()

    def reasons(self) -> tuple[str, ...]:
        return tuple(item.describe() for item in self.restored)


class CustodySet:
    """The facts one turn is holding, and what happened to them."""

    def __init__(self, turn_id: str) -> None:
        self.turn_id = str(turn_id)
        self.started_at = time.time()
        self._lock = checked_lock("fact_custody_set", reentrant=True)
        self._facts: dict[tuple[str, str], HeldFact] = {}
        self._breaks: list[CustodyBreak] = []
        self._checkpoints = 0
        self._dropped_facts = 0

    # ── holding ────────────────────────────────────────────────────────────

    def hold(self, fact: HeldFact) -> HeldFact:
        """Take custody of a fact. A better-graded fact replaces a weaker one.

        Same subject and predicate arriving twice is normal — a recall pass and
        a live read can both establish the count. The stronger evidence wins,
        and a tie keeps the incumbent so that a re-derivation cannot quietly
        restate an older reading as though it were new.
        """

        with self._lock:
            existing = self._facts.get(fact.key)
            if existing is not None and existing.grade >= fact.grade:
                return existing
            if existing is None and len(self._facts) >= _MAX_FACTS:
                self._dropped_facts += 1
                logger.debug(
                    "fact custody full at %d; not holding %s.%s",
                    _MAX_FACTS,
                    fact.subject,
                    fact.predicate,
                )
                return fact
            self._facts[fact.key] = fact
            return fact

    def facts(self) -> tuple[HeldFact, ...]:
        with self._lock:
            return tuple(self._facts.values())

    # ── checking ───────────────────────────────────────────────────────────

    def inspect(
        self,
        stage: str,
        before: Any,
        after: Any,
        *,
        emit_log: bool = True,
    ) -> tuple[CustodyBreak, ...]:
        """Check one stage's rewrite against everything the turn is holding.

        Called from the seam every text mutation already passes through, so a
        new stage is covered the day it is written rather than the day someone
        remembers to instrument it.
        """

        before_text = str(before or "")
        after_text = str(after or "")
        if before_text == after_text:
            return ()
        found: list[CustodyBreak] = []
        with self._lock:
            self._checkpoints += 1
            for fact in self._facts.values():
                # Contradiction is tested first, and it is not an ordering
                # preference. A stage that replaces the true sentence with a
                # false one has done both things at once: the fact is absent
                # AND a competing value is stated. Reported as a drop, the
                # repair appends the truth underneath the falsehood and leaves
                # the person holding two numbers. Reported as a contradiction,
                # the offending sentence is the thing that gets replaced.
                contradiction = fact.contradicting_sentence(after_text)
                if contradiction and not fact.contradicting_sentence(before_text):
                    found.append(
                        CustodyBreak(
                            fact=fact,
                            stage=str(stage or "unknown"),
                            kind=BreakKind.CONTRADICTED,
                            detail=contradiction[:200],
                        )
                    )
                    continue
                if fact.present_in(before_text) and not fact.present_in(after_text):
                    found.append(
                        CustodyBreak(
                            fact=fact,
                            stage=str(stage or "unknown"),
                            kind=BreakKind.DROPPED,
                            detail=(
                                f"stated before this stage ({len(before_text)} chars) "
                                f"and absent after ({len(after_text)} chars)"
                            ),
                        )
                    )
            self._breaks.extend(found)
            if len(self._breaks) > _MAX_BREAKS:
                del self._breaks[_MAX_BREAKS:]
        if emit_log:
            for item in found:
                logger.warning("🔗 [CUSTODY] %s", item.describe())
        return tuple(found)

    def breaks(self) -> tuple[CustodyBreak, ...]:
        with self._lock:
            return tuple(self._breaks)

    def live_breaks(self, text: Any) -> tuple[CustodyBreak, ...]:
        """Breaks still true of this text.

        A fact a later stage restated is no longer broken, whatever happened in
        the middle. Custody is a property of what gets delivered, not a grudge
        about the pipeline.
        """

        rendered = str(text or "")
        live: list[CustodyBreak] = []
        seen: set[tuple[str, str]] = set()
        for item in self.breaks():
            if item.fact.key in seen:
                continue
            if item.fact.present_in(rendered):
                continue
            if item.kind is BreakKind.CONTRADICTED and not item.fact.contradicting_sentence(
                rendered
            ):
                continue
            seen.add(item.fact.key)
            live.append(item)
        return tuple(live)

    # ── enforcing ──────────────────────────────────────────────────────────

    def restore(self, text: Any, *, emit_log: bool = True) -> Restoration:
        """Put back what the pipeline lost, on the exact text about to be sent.

        A contradicted fact has its offending sentence REPLACED by the held
        rendering rather than having a correction appended after it. Two
        sentences disagreeing about the same count, one of them true, is not an
        improvement on one sentence that is false — the person has no way to
        know which to believe.
        """

        rendered = str(text or "")
        restored: list[CustodyBreak] = []
        unrestored: list[CustodyBreak] = []
        for item in self.live_breaks(rendered):
            if not item.fact.restorable:
                unrestored.append(item)
                continue
            replacement = item.fact.canonical_rendering.strip()
            if not replacement:
                unrestored.append(item)
                continue
            if item.kind is BreakKind.CONTRADICTED:
                offending = item.fact.contradicting_sentence(rendered)
                if offending and offending in rendered:
                    rendered = rendered.replace(offending, replacement, 1)
                    restored.append(item)
                    continue
                # The sentence moved under us between detection and repair.
                # Appending is still strictly better than shipping the wrong
                # number unanswered, and the break stays on the record either
                # way.
            rendered = (rendered.rstrip() + "\n\n" + replacement).strip()
            restored.append(item)
        if restored and emit_log:
            logger.warning(
                "🔗 [CUSTODY] restored %d fact(s) the pipeline lost: %s",
                len(restored),
                "; ".join(item.describe() for item in restored)[:400],
            )
        return Restoration(
            text=rendered,
            changed=bool(restored),
            restored=tuple(restored),
            unrestored=tuple(unrestored),
        )

    # ── reporting ──────────────────────────────────────────────────────────

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "turn_id": self.turn_id,
                "held": len(self._facts),
                "checkpoints": self._checkpoints,
                "breaks": [item.to_dict() for item in self._breaks[:16]],
                "break_count": len(self._breaks),
                "facts_not_held_over_capacity": self._dropped_facts,
                "stages_that_broke_custody": sorted(
                    {item.stage for item in self._breaks}
                ),
            }


_REGISTRY_LOCK = checked_lock("fact_custody.registry", rank=LockRank.LEAF, reentrant=True)
_SETS: dict[str, CustodySet] = {}


def custody_for(turn_id: str) -> CustodySet:
    """The custody set for a turn, created on first use and bounded."""

    key = str(turn_id or "unknown")
    with _REGISTRY_LOCK:
        existing = _SETS.get(key)
        if existing is not None:
            return existing
        created = CustodySet(key)
        _SETS[key] = created
        if len(_SETS) > _MAX_TURNS:
            for stale in sorted(_SETS, key=lambda k: _SETS[k].started_at)[
                : len(_SETS) - _MAX_TURNS
            ]:
                _SETS.pop(stale, None)
        return created


def current_custody() -> CustodySet | None:
    """The custody set of the turn bound to this context, if there is one.

    Keyed off the TurnOutcome's own turn id rather than a second binding of its
    own. Two independent notions of "the current turn" is how a ledger ends up
    describing a different turn than the one being served.
    """

    outcome = current_turn()
    if outcome is None:
        return None
    return custody_for(outcome.turn_id)


def hold_fact(
    *,
    subject: str,
    predicate: str,
    value: str,
    canonical_rendering: str,
    established_by: str,
    subject_cues: tuple[str, ...] | list[str] = (),
    grade: VerificationGrade = VerificationGrade.ASSERTED,
    kind: ValueKind = ValueKind.TEXT,
    value_forms: tuple[str, ...] | list[str] = (),
    evidence: tuple[str, ...] | list[str] = (),
) -> HeldFact | None:
    """Take custody of a fact on the current turn.

    Returns None when no turn is bound — a background tick has no reply whose
    custody could be broken, and inventing a ledger for it would collect facts
    nothing will ever check.
    """

    custody = current_custody()
    if custody is None:
        return None
    fact = HeldFact(
        subject=str(subject),
        predicate=str(predicate),
        value=str(value),
        subject_cues=tuple(str(cue) for cue in subject_cues if str(cue).strip()),
        canonical_rendering=str(canonical_rendering),
        established_by=str(established_by),
        grade=grade,
        kind=kind,
        value_forms=tuple(str(form) for form in value_forms if str(form).strip()),
        evidence=tuple(str(item) for item in evidence)[:8],
    )
    return custody.hold(fact)


def inspect_mutation(
    stage: str,
    before: Any,
    after: Any,
    *,
    emit_log: bool = True,
) -> tuple[CustodyBreak, ...]:
    """Check a stage's rewrite against the current turn's custody set."""

    custody = current_custody()
    if custody is None:
        return ()
    return custody.inspect(stage, before, after, emit_log=emit_log)


def restore_held_facts(text: Any, *, emit_log: bool = True) -> Restoration:
    """Enforce custody on outgoing text. Safe to call with no turn bound."""

    custody = current_custody()
    if custody is None:
        return Restoration(text=str(text or ""), changed=False)
    return custody.restore(text, emit_log=emit_log)


def custody_report(turn_id: str | None = None) -> dict[str, Any]:
    """What custody knows, for the health report and for a live investigation."""

    if turn_id:
        with _REGISTRY_LOCK:
            found = _SETS.get(str(turn_id))
        return found.report() if found is not None else {"turn_id": turn_id, "held": 0}
    with _REGISTRY_LOCK:
        sets = sorted(_SETS.values(), key=lambda s: s.started_at, reverse=True)
    recent = [item.report() for item in sets[:8]]
    return {
        "turns_tracked": len(sets),
        "turns_with_breaks": sum(1 for item in sets if item.breaks()),
        "recent": recent,
        "stages_that_broke_custody": sorted(
            {stage for item in recent for stage in item["stages_that_broke_custody"]}
        ),
    }


def forget_custody(turn_id: str) -> None:
    with _REGISTRY_LOCK:
        _SETS.pop(str(turn_id), None)


def reset_custody_for_test() -> None:
    with _REGISTRY_LOCK:
        _SETS.clear()
