"""Noticing that she has stopped getting anywhere.

Two narrow guards already existed. `environment/action_gateway` suppresses an
action that has failed twice with the same context, and `perception/action_gateway`
blocks one that recently failed with high surprise. Both catch the same single
shape — *repeated failure* — inside a single gateway, and both answer only by
vetoing the next action.

Four other shapes were invisible:

* the same action **succeeding** identically forever (`ls` in a loop returns
  0 and looks healthy every time),
* **oscillation** — edit, test, edit, test, edit, test, with neither ever
  changing anything,
* **monologue** — consecutive messages with no tool call and no new input,
* repeated **context-window overflow**, which is a memory-management failure
  wearing a model error's clothes.

None of these are failures at the step level. Every individual step succeeds.
That is exactly why nothing caught them: a per-action gate cannot see a pattern
that only exists across steps.

**The false positive this is designed around.** The prior art (OpenHands, MIT)
shipped loop detection and then had to fix it, because it killed agents that
were legitimately polling a long-running process — a build, a test run, a
deploy. Waiting *is* repeating, and it is also correct. So `kind="wait"` steps
are excluded from repeat detection outright, and any step carrying a
``progress_marker`` that has changed is not a repeat however identical the rest
of it looks. Learning the lesson from someone else's incident is the entire
point of reading their code; shipping their bug first would waste it.

**Recovery is a ladder, not a switch.** Their other fix was replacing a hard
error state — from which the agent could not be talked back — with a graceful
transition. A detector that can only halt turns a recoverable rut into a dead
session, so the verdict names the pattern and the evidence, and the remedy
escalates: nudge, then force a change of strategy, then ask the human.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterable, Sequence

logger = logging.getLogger("Aura.StuckDetector")

__all__ = [
    "StuckPattern",
    "AgentStep",
    "StuckVerdict",
    "Remedy",
    "StuckDetector",
]

#: Volatile substrings stripped before comparison. A tool result that differs
#: only by a timestamp or an object id is the same result.
_VOLATILE = re.compile(
    r"0x[0-9a-fA-F]{6,}"                      # object ids
    r"|\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\S*"  # ISO timestamps
    r"|\b\d{10,}\b"                           # epoch-ish integers
    r"|\bpid[= ]\d+",                          # pids
)


class StuckPattern(StrEnum):
    REPEATED_ACTION_OBSERVATION = "repeated_action_observation"
    REPEATED_ACTION_ERROR = "repeated_action_error"
    MONOLOGUE = "monologue"
    OSCILLATION = "oscillation"
    REPEATED_CONTEXT_OVERFLOW = "repeated_context_overflow"
    #: Different actions, unchanged world.
    #:
    #: The one shape that is not a repetition: she *varied* what she tried and
    #: the observation never moved. Plain repetition is a stuck hand; this is a
    #: stuck world, and it is the only pattern here that a caller cannot fix by
    #: telling her to stop doing the same thing.
    NO_PROGRESS = "no_progress"


class Remedy(StrEnum):
    """Escalating responses. A detector that can only halt kills a session."""

    NONE = "none"
    NUDGE = "nudge"                    # tell her she is repeating; let her adjust
    FORCE_NEW_STRATEGY = "force_new_strategy"  # constrain away the repeated move
    # Two ways out that are not "try harder at the thing that is not working".
    #
    # A run that has learned something mid-task can be worth abandoning: the
    # position is bad, the knowledge is new, and the next attempt starts from
    # both. A run can equally be worth finishing badly — the ending is where
    # the evidence is, and a task that never reaches one teaches nothing about
    # how it goes wrong.
    #
    # Both are choices about the task rather than moves inside it, and having
    # neither is what leaves a loop with only "keep pressing" and "stop".
    START_OVER = "start_over"          # abandon this attempt, begin again knowing more
    SEE_IT_THROUGH = "see_it_through"  # finish it badly on purpose, and learn from the ending
    ASK_HUMAN = "ask_human"            # surface it; she is not getting out alone


@dataclass(frozen=True)
class AgentStep:
    """One action and what came back.

    ``kind`` distinguishes a tool call from a message from a deliberate wait.
    ``progress_marker`` is any caller-supplied token that legitimately advances
    while the action stays identical — a build's line count, a queue depth, a
    poll's elapsed time. Its whole job is to keep honest waiting from reading
    as a loop.
    """

    action: str
    arguments: str = ""
    observation: str = ""
    is_error: bool = False
    kind: str = "tool"
    progress_marker: str | None = None

    @property
    def is_wait(self) -> bool:
        return self.kind == "wait"

    def signature(self) -> tuple[str, str]:
        """What the agent *did*, ignoring volatile detail."""
        return (self.action.strip(), _normalize(self.arguments))

    def outcome(self) -> str:
        """What came *back*, ignoring volatile detail."""
        return _normalize(self.observation)

    def fingerprint(self) -> tuple[str, str, str, str | None]:
        """Everything that decides whether two steps are 'the same'.

        The progress marker is part of the fingerprint on purpose: two polls of
        a running build are only the same step if the build has not moved.
        """
        action, arguments = self.signature()
        return (action, arguments, self.outcome(), self.progress_marker)


def _normalize(text: str) -> str:
    return _VOLATILE.sub("<v>", (text or "").strip())


@dataclass(frozen=True)
class StuckVerdict:
    """The finding, with the evidence attached.

    A bare "you are stuck" cannot be acted on — not by the model, not by a
    human reading a log. The pattern says what shape the rut is and the
    evidence says which steps proved it.
    """

    stuck: bool = False
    pattern: StuckPattern | None = None
    remedy: Remedy = Remedy.NONE
    evidence: tuple[str, ...] = ()
    detail: str = ""

    def __bool__(self) -> bool:
        return self.stuck

    def describe(self) -> str:
        if not self.stuck:
            return "not stuck"
        return f"{self.pattern}: {self.detail}"


class StuckDetector:
    """Rule-based detection of a run that has stopped going anywhere.

    Deliberately not a model call. This runs on every step, it has to be
    trustworthy while the thing it watches is misbehaving, and a detector that
    needs the cortex cannot report that the cortex is the problem.
    """

    def __init__(
        self,
        *,
        window: int = 24,
        repeat_threshold: int = 4,
        error_threshold: int = 3,
        monologue_threshold: int = 3,
        oscillation_cycles: int = 3,
        no_progress_threshold: int = 3,
    ) -> None:
        if window < 2:
            raise ValueError("window must be at least 2")
        for name, value in (
            ("repeat_threshold", repeat_threshold),
            ("error_threshold", error_threshold),
            ("monologue_threshold", monologue_threshold),
            ("oscillation_cycles", oscillation_cycles),
            ("no_progress_threshold", no_progress_threshold),
        ):
            if value < 2:
                raise ValueError(f"{name} must be at least 2 to describe a repetition")
        self.window = window
        self.repeat_threshold = repeat_threshold
        self.error_threshold = error_threshold
        self.monologue_threshold = monologue_threshold
        self.oscillation_cycles = oscillation_cycles
        self.no_progress_threshold = no_progress_threshold
        #: How many times a rut has already been called on this run, so the
        #: remedy can escalate rather than repeating a nudge that did not work.
        self._interventions = 0

    # -- public ------------------------------------------------------------

    def check(self, steps: Sequence[AgentStep]) -> StuckVerdict:
        """Look at the recent history and decide whether she is going in circles."""
        found = self.check_all(steps)
        if not found:
            return StuckVerdict()
        self._interventions += 1
        escalated = self._escalate(found[0])
        logger.info("stuck: %s", escalated.describe())
        return escalated

    def check_all(self, steps: Sequence[AgentStep]) -> list[StuckVerdict]:
        """Every pattern currently visible, strongest first, without escalating.

        A window can hold more than one rut at a time — a caller that already
        acknowledged the first still needs to hear about a genuinely different
        second one, and a single-verdict interface can only answer "the same
        thing you already knew". Ordering is the priority order below and does
        not change; only the count does.

        Escalation is deliberately not applied here. Reading the state must not
        advance the remedy ladder, or a caller that inspects twice gets a
        harsher answer for looking.
        """
        recent = list(steps)[-self.window:]
        if len(recent) < 2:
            return []
        found: list[StuckVerdict] = []
        for detect in (
            self._context_overflow,
            self._repeated_error,
            self._repeated_observation,
            self._no_progress,
            self._oscillation,
            self._monologue,
        ):
            # The repetition detectors return every fingerprint at or above
            # threshold, because a window can hold two different repeated
            # actions and reporting only the most common one hides the second
            # behind the first. The rest describe a single shape and return one
            # verdict. Both arrive here as a sequence.
            result = detect(recent)
            verdicts = result if isinstance(result, list) else [result]
            found.extend(v for v in verdicts if v.stuck)
        return found

    def escalate(self, verdict: StuckVerdict) -> StuckVerdict:
        """Count this finding as an intervention and attach the remedy it earns.

        Public because a caller using :meth:`check_all` to pick which finding to
        act on still has to advance the ladder for the one it chose — and doing
        that by reaching into the counter would put the escalation policy in two
        places.
        """
        self._interventions += 1
        return self._escalate(verdict)

    def _escalate(self, verdict: StuckVerdict) -> StuckVerdict:
        return StuckVerdict(
            stuck=True,
            pattern=verdict.pattern,
            remedy=self._remedy_for(verdict.pattern),
            evidence=verdict.evidence,
            detail=verdict.detail,
        )

    def reset(self) -> None:
        """Forget prior interventions. Call when real progress resumes."""
        self._interventions = 0

    @property
    def interventions(self) -> int:
        return self._interventions

    def _remedy_for(self, pattern: StuckPattern | None) -> Remedy:
        # Context overflow is never something she can talk her way out of: the
        # window is full and the next call fails the same way.
        if pattern is StuckPattern.REPEATED_CONTEXT_OVERFLOW:
            return Remedy.FORCE_NEW_STRATEGY if self._interventions < 2 else Remedy.ASK_HUMAN
        # No progress is the other pattern a nudge cannot touch. "Stop repeating
        # yourself" is the wrong advice to someone who already varied what they
        # tried — by the time this fires, trying something else is the thing that
        # has been failing.
        if pattern is StuckPattern.NO_PROGRESS:
            return Remedy.FORCE_NEW_STRATEGY if self._interventions < 2 else Remedy.ASK_HUMAN
        # One nudge, then escalate. Repeating a nudge that already failed to
        # change anything is itself a loop, which would be a poor look for the
        # loop detector. _interventions is >= 1 here: check() increments before
        # asking.
        if self._interventions == 1:
            return Remedy.NUDGE
        if self._interventions == 2:
            return Remedy.FORCE_NEW_STRATEGY
        return Remedy.ASK_HUMAN

    # -- patterns ----------------------------------------------------------

    @staticmethod
    def _actionable(steps: Iterable[AgentStep]) -> list[AgentStep]:
        """Steps that can meaningfully repeat *as actions*.

        Waits are dropped here, once, rather than special-cased in each
        detector — polling a build is repetition and is also correct, and that
        distinction should live in one place.

        Messages are dropped for a sharper reason. They carry no action, so a
        run of them is three steps with an identical empty fingerprint, and the
        action-repetition detectors fired on it before the monologue detector
        was ever reached: three sentences in a row were reported as "'' ran 3
        times with an identical result". A monologue is a real pattern and it
        already has its own detector; letting it also register as a repeated
        action means the verdict names the wrong problem and the remedy treats
        the wrong thing.
        """
        return [s for s in steps if not s.is_wait and s.kind != "message"]

    def _repeated_observation(self, steps: Sequence[AgentStep]) -> list[StuckVerdict]:
        """The same action returning the same thing, over and over.

        Every fingerprint at or above threshold, not just the most common one.
        Two different actions can both be looping in one window, and
        ``most_common(1)`` would report whichever appeared first and hide the
        other for as long as it stayed in the window.
        """
        candidates = [s for s in self._actionable(steps) if not s.is_error]
        if len(candidates) < self.repeat_threshold:
            return []

        counts = Counter(s.fingerprint() for s in candidates)
        return [
            StuckVerdict(
                stuck=True,
                pattern=StuckPattern.REPEATED_ACTION_OBSERVATION,
                evidence=(fingerprint[0],),
                detail=(
                    f"{fingerprint[0]!r} ran {count} times with an identical result "
                    "and nothing changed"
                ),
            )
            for fingerprint, count in counts.most_common()
            if count >= self.repeat_threshold
        ]

    def _repeated_error(self, steps: Sequence[AgentStep]) -> list[StuckVerdict]:
        """The same action failing the same way. Every such action, not one."""
        candidates = [s for s in self._actionable(steps) if s.is_error]
        if len(candidates) < self.error_threshold:
            return []

        counts = Counter(s.fingerprint() for s in candidates)
        found: list[StuckVerdict] = []
        for fingerprint, count in counts.most_common():
            if count < self.error_threshold:
                continue
            # The error itself goes in the detail. "Failed 3 times with the same
            # error" tells a reader there is a loop and not what the loop is
            # about, which is the half that decides whether it is fixable.
            failure = fingerprint[2]
            found.append(
                StuckVerdict(
                    stuck=True,
                    pattern=StuckPattern.REPEATED_ACTION_ERROR,
                    evidence=(fingerprint[0],),
                    detail=(
                        f"{fingerprint[0]!r} failed {count} times with the same error"
                        + (f" ({failure})" if failure else "")
                        + "; retrying it again will fail the same way"
                    ),
                )
            )
        return found

    def _no_progress(self, steps: Sequence[AgentStep]) -> StuckVerdict:
        """She varied what she tried and the world did not move.

        Checked after plain repetition and before oscillation. Repetition is the
        more specific finding — if the actions were also identical, that is a
        stuck hand and says so. This only fires once *more than one* distinct
        action has produced the same unchanging observation, which is what makes
        it a statement about the environment rather than about her.
        """
        candidates = [s for s in self._actionable(steps) if not s.is_error]
        if len(candidates) < self.no_progress_threshold:
            return StuckVerdict()

        tail = candidates[-self.no_progress_threshold:]
        outcomes = {s.outcome() for s in tail}
        actions = {s.signature() for s in tail}
        if len(outcomes) != 1 or len(actions) < 2:
            return StuckVerdict()

        # An absent observation is no evidence about the world, not evidence
        # that the world is unchanged. Without this, a healthy turn running
        # perceive → recall → reason → answer — four different phases, none of
        # which records an observation — matched "different actions, identical
        # outcome" perfectly and was reported as a stuck world. That is the
        # false positive this whole module is designed around, arriving through
        # the newest pattern.
        if not next(iter(outcomes)).strip():
            return StuckVerdict()

        return StuckVerdict(
            stuck=True,
            pattern=StuckPattern.NO_PROGRESS,
            evidence=tuple(sorted({a for a, _ in actions})),
            detail=(
                f"{len(actions)} different actions over {len(tail)} steps and the "
                "observation never changed; the state is not moving"
            ),
        )

    def _oscillation(self, steps: Sequence[AgentStep]) -> StuckVerdict:
        """A, B, A, B — two moves that undo or ignore each other."""
        candidates = self._actionable(steps)
        needed = self.oscillation_cycles * 2
        if len(candidates) < needed:
            return StuckVerdict()

        tail = candidates[-needed:]
        evens = {s.fingerprint() for s in tail[0::2]}
        odds = {s.fingerprint() for s in tail[1::2]}
        if len(evens) != 1 or len(odds) != 1 or evens == odds:
            return StuckVerdict()

        first, second = next(iter(evens)), next(iter(odds))
        return StuckVerdict(
            stuck=True,
            pattern=StuckPattern.OSCILLATION,
            evidence=(first[0], second[0]),
            detail=(
                f"alternating between {first[0]!r} and {second[0]!r} for "
                f"{self.oscillation_cycles} cycles without either changing anything"
            ),
        )

    def _monologue(self, steps: Sequence[AgentStep]) -> StuckVerdict:
        """Consecutive messages with no action and no new input."""
        trailing = 0
        for step in reversed(steps):
            if step.kind != "message":
                break
            trailing += 1
        if trailing < self.monologue_threshold:
            return StuckVerdict()

        return StuckVerdict(
            stuck=True,
            pattern=StuckPattern.MONOLOGUE,
            evidence=tuple(s.action for s in steps[-trailing:]),
            detail=(
                f"{trailing} consecutive messages with no tool call and no new "
                "input — talking rather than working"
            ),
        )

    def _context_overflow(self, steps: Sequence[AgentStep]) -> StuckVerdict:
        """Repeated context-window errors.

        Checked first, and separately from ordinary repeated errors, because
        the remedy is different in kind: no amount of rephrasing helps, the
        window has to be made smaller.
        """
        overflows = [
            s for s in steps
            if s.is_error and "context" in s.observation.lower()
            and any(w in s.observation.lower() for w in ("window", "length", "token"))
        ]
        if len(overflows) < 2:
            return StuckVerdict()

        return StuckVerdict(
            stuck=True,
            pattern=StuckPattern.REPEATED_CONTEXT_OVERFLOW,
            evidence=tuple(dict.fromkeys(s.action for s in overflows)),
            detail=(
                f"{len(overflows)} context-window errors; the window must be "
                "condensed, not the request retried"
            ),
        )
