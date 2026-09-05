"""Loop detection reported as a no-change impasse.

This module used to carry a second, independent implementation of the same five
patterns that :mod:`core.runtime.stuck_detector` detects. Two detectors, two
sets of thresholds, two windows — 3 and 20 here, 4 and 24 there — and two
consumers that could never agree on whether a turn was stuck. Each had a pattern
the other lacked, so neither was simply the better one: this side knew about
*no progress* (varied actions, unchanged world), and the runtime side knew about
repeated context overflow, carried an escalating remedy, stripped volatile
detail before comparing, and had the ``progress_marker`` guard that stops a long
poll being killed as a loop.

That is now one detector. The mechanism lives in :mod:`core.runtime.stuck_detector`
— the foundation layer, which :mod:`core.observability` may import and this
module may too — with ``NO_PROGRESS`` moved into it. What is left here is the
one thing that genuinely cannot live down there: the impasse.

Why the split is real and not bureaucratic
------------------------------------------
``core/runtime`` may not import cognition; ``core/observability`` may not import
agency. So a detector that constructs a
:class:`~core.cognition.impasse.Impasse` cannot be the detector that
``turn_observer`` uses. Rather than duplicate detection to satisfy that, the
detection is shared and only the *reporting* differs — which is the honest
factoring anyway. Detecting a loop and deciding what a loop means to the rest of
cognition are different jobs.

Why it reports an impasse
-------------------------
Soar's :attr:`~core.cognition.impasse.ImpasseType.NO_CHANGE` is "something was
selected and applying it changed nothing", which is exactly what a loop is.
Saying so puts loop detection into machinery that already exists:
:class:`~core.cognition.impasse.ImpasseLearner` counts impasses by type, so the
loop rate becomes a reportable diagnostic beside every other way a decision can
fail, instead of a log line nobody aggregates.

Thresholds
----------
The adapter passes its own thresholds rather than taking the runtime defaults.
Three, not four, for a repeat: three is the first count containing *two*
consecutive repeats of the same transition, and one retry after a transient
failure is correct behaviour a detector must not punish. That rationale was
written for the tool loop this module serves, and keeping it here means sharing
the mechanism did not silently recalibrate the caller.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.cognition.impasse import Impasse, ImpasseType, situation_signature
from core.runtime.stuck_detector import AgentStep as _RuntimeStep
from core.runtime.stuck_detector import Remedy, StuckPattern
from core.runtime.stuck_detector import StuckDetector as _Mechanism

__all__ = [
    "StuckPattern",
    "Remedy",
    "AgentStep",
    "StuckVerdict",
    "StuckDetector",
    "digest_of",
]

#: Repeats of an identical action-and-outcome before it counts as a loop.
#:
#: Three occurrences, because three is the first count containing *two*
#: consecutive repeats of the same transition. Two occurrences is one retry, and
#: retrying once after a transient failure is correct behaviour that a detector
#: must not call a loop.
_REPEAT_THRESHOLD = 3

#: Steps over which the observation never changes, with more than one distinct
#: action attempted, before the world is called unmoved.
_NO_PROGRESS_THRESHOLD = 3

#: Consecutive turns that produced words and no action.
_MONOLOGUE_THRESHOLD = 3

#: Full A-B-A-B cycles before alternation counts.
_OSCILLATION_CYCLES = 2

#: Most recent steps considered. A window rather than the whole history because
#: an agent that looped, recovered, and moved on is not stuck now, and a
#: detector with unbounded memory would keep reporting a resolved loop.
DEFAULT_WINDOW = 20


def digest_of(value: Any) -> str:
    """A stable short digest of any JSON-ish value.

    Observations are digested rather than stored. The detector only ever asks
    whether two observations are the same, and keeping the payloads would mean a
    loop-detection window holding twenty tool outputs — including whatever they
    contained — for the lifetime of the loop.
    """
    if value is None:
        return ""
    try:
        text = json.dumps(value, sort_keys=True, default=repr)
    except (TypeError, ValueError):
        text = repr(value)
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


class AgentStep(_RuntimeStep):
    """One thing the agent did, and what came back, with payloads digested.

    A subclass rather than a parallel type so the shared mechanism compares
    these with the same fingerprint logic it uses everywhere else. Digesting
    happens in :meth:`of`; the mechanism never sees the payload, and normalising
    a digest is a no-op, so the volatile-stripping upstream costs nothing here.
    """

    @staticmethod
    def of(
        action: str,
        *,
        arguments: Any = None,
        observation: Any = None,
        failed: bool = False,
        error_kind: str = "",
        kind: str = "tool",
        progress_marker: str | None = None,
    ) -> AgentStep:
        # A failed step is identified by how it failed, not by what it returned.
        # Folding error_kind into the observation is what makes "the same action
        # failing the same way" a distinct fingerprint from "the same action
        # failing differently", which is the distinction the error pattern rests
        # on.
        outcome = str(error_kind) if failed else digest_of(observation)
        return AgentStep(
            action=str(action),
            arguments=digest_of(arguments),
            observation=outcome,
            is_error=bool(failed),
            kind=kind,
            progress_marker=progress_marker,
        )

    @property
    def call_key(self) -> str:
        """Identity of the call: what was done, with what."""
        return f"{self.action}#{self.arguments}"

    @property
    def cycle_key(self) -> str:
        """Identity of the whole transition: the call and what it produced."""
        return f"{self.call_key}->{self.observation}"

    @property
    def failed(self) -> bool:
        """Whether the step errored. Named for the tool loop that reads it."""
        return self.is_error

    @property
    def error_kind(self) -> str:
        """How it failed, or empty for a step that did not.

        The error kind *is* the observation for a failed step — see :meth:`of`,
        where folding it there is what makes "failed the same way" a different
        fingerprint from "failed differently".
        """
        return self.observation if self.is_error else ""


@dataclass(frozen=True)
class StuckVerdict:
    """A detected loop, the evidence for it, and the impasse it constitutes."""

    pattern: StuckPattern
    detail: str
    repetitions: int
    actions: tuple[str, ...]
    impasse: Impasse
    remedy: Remedy = Remedy.NONE

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern.value,
            "detail": self.detail,
            "repetitions": self.repetitions,
            "actions": list(self.actions),
            "remedy": self.remedy.value,
            "impasse": self.impasse.type.value,
            "signature": self.impasse.signature,
        }


@dataclass
class StuckDetector:
    """Watches a stream of steps and says when it has stopped going anywhere.

    One detector per agent loop. :meth:`reset` is called when a new instruction
    arrives, because a user turn is what makes prior repetition irrelevant — the
    agent may legitimately be asked to do the same thing again.
    """

    scope: str = "agent_loop"
    window: int = DEFAULT_WINDOW
    _steps: deque[AgentStep] = field(default_factory=deque)
    _reported: set[str] = field(default_factory=set)
    _mechanism: _Mechanism = field(init=False, repr=False)

    def __post_init__(self) -> None:
        needed = _OSCILLATION_CYCLES * 2
        if self.window < needed:
            raise ValueError(
                f"window must hold at least {needed} steps to detect alternation"
            )
        self._steps = deque(self._steps, maxlen=self.window)
        self._mechanism = _Mechanism(
            window=self.window,
            repeat_threshold=_REPEAT_THRESHOLD,
            error_threshold=_REPEAT_THRESHOLD,
            monologue_threshold=_MONOLOGUE_THRESHOLD,
            oscillation_cycles=_OSCILLATION_CYCLES,
            no_progress_threshold=_NO_PROGRESS_THRESHOLD,
        )

    # -- recording -------------------------------------------------------

    def reset(self) -> None:
        """Forget the window. Called when the user says something new."""
        self._steps.clear()
        self._reported.clear()
        self._mechanism.reset()

    def observe(self, step: AgentStep) -> None:
        self._steps.append(step)

    def observe_idle_turn(self) -> None:
        """Record a turn that produced words and no action."""
        self._steps.append(AgentStep(action="", kind="message"))

    @property
    def steps(self) -> tuple[AgentStep, ...]:
        return tuple(self._steps)

    @property
    def interventions(self) -> int:
        return self._mechanism.interventions

    # -- detection -------------------------------------------------------

    def assess(self, *, context: Mapping[str, Any] | None = None) -> StuckVerdict | None:
        """The strongest loop currently visible, or None."""
        verdict = self._mechanism.check(tuple(self._steps))
        if not verdict.stuck or verdict.pattern is None:
            return None
        return self._as_impasse(verdict, context)

    def _as_impasse(self, verdict: Any, context: Mapping[str, Any] | None) -> StuckVerdict:
        ctx = dict(context or {})
        ctx.setdefault("scope", self.scope)
        actions = tuple(verdict.evidence)
        return StuckVerdict(
            pattern=verdict.pattern,
            detail=verdict.detail,
            repetitions=self._repetitions(verdict.pattern, actions),
            actions=actions,
            remedy=verdict.remedy,
            impasse=Impasse(
                type=ImpasseType.NO_CHANGE,
                signature=situation_signature(ctx, actions or (verdict.pattern.value,)),
                candidates=actions,
                detail=verdict.detail,
            ),
        )

    def _repetitions(self, pattern: StuckPattern, actions: Sequence[str]) -> int:
        """How many recent steps the pattern actually spans.

        Counted from the window rather than reported by the mechanism, which
        deals in verdicts rather than tallies. A caller deciding whether to
        escalate wants to know how deep the rut is.
        """
        if not actions:
            return 0
        wanted = set(actions)
        return sum(1 for step in self._steps if step.action in wanted)

    def assess_once(self, *, context: Mapping[str, Any] | None = None) -> StuckVerdict | None:
        """:meth:`assess`, but each distinct loop is reported only once.

        A caller acting on a verdict does not necessarily clear the window, so
        the same loop would be re-reported on every subsequent step and a single
        stuck episode would look like a dozen. The signature is what identifies
        an episode, so a genuinely new loop still reports.
        """
        for raw in self._mechanism.check_all(tuple(self._steps)):
            if raw.pattern is None:
                continue
            verdict = self._as_impasse(raw, context)
            key = f"{verdict.pattern.value}:{verdict.impasse.signature}"
            if key in self._reported:
                # Already acknowledged. Keep looking rather than returning None:
                # a window can hold two different ruts, and reporting only the
                # first would hide a genuinely new one behind an old one that
                # has not aged out yet.
                continue
            self._reported.add(key)
            return StuckVerdict(
                pattern=verdict.pattern,
                detail=verdict.detail,
                repetitions=verdict.repetitions,
                actions=verdict.actions,
                remedy=self._mechanism.escalate(raw).remedy,
                impasse=verdict.impasse,
            )
        return None

    def report(self) -> dict[str, Any]:
        """What this detector is holding, for a health surface to read."""
        return {
            "scope": self.scope,
            "window": self.window,
            "steps_held": len(self._steps),
            "loops_reported": len(self._reported),
            "interventions": self._mechanism.interventions,
        }

    def record_to_learner(self, verdict: StuckVerdict) -> None:
        """File the verdict with the process-wide impasse learner.

        Separate from detection so that a caller can assess without recording —
        a probe, a dry run, a test — and so the import stays out of the hot path
        for callers that only want the verdict.
        """
        from core.cognition.impasse import get_impasse_learner

        get_impasse_learner().record_impasse(verdict.impasse)


def steps_from(records: Sequence[Mapping[str, Any]]) -> list[AgentStep]:
    """Build a step window from stored tool records.

    Records with no action name are dropped rather than turned into a step with
    an empty action. Several of those in a row are identical by fingerprint and
    would read as a loop of nothing.
    """
    steps: list[AgentStep] = []
    for record in records:
        name = str(record.get("name") or record.get("tool") or record.get("action") or "").strip()
        if not name:
            continue
        error = record.get("error")
        steps.append(
            AgentStep.of(
                name,
                arguments=record.get("arguments"),
                observation=record.get("result"),
                failed=bool(error),
                error_kind=str(record.get("error_kind") or error or ""),
            )
        )
    return steps
