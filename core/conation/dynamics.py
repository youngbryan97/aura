"""core/conation/dynamics.py — what a motive does over time.

A motivational state that does not change is not a motive, it is a
preference. Three temporal behaviours separate the two, and a system missing
any of them behaves in a way that is recognisably wrong.

## Satiation

Curiosity that does not fall when the thing is understood is not curiosity.
It is a permanent behavioural bias named ``explore``, and it produces an agent
that inspects the same solved object forever because inspection scored well
last time. The same holds everywhere: a want that survives its own
satisfaction undamaged is a leak.

    want -> attain -> satisfy -> decay

Consummation suppresses the motive it consumed, in proportion to how
completely it was consumed. The suppression then recovers, because a body
gets hungry again, and the recovery rate is what gives behaviour its rhythm.
An agent with satiation but no recovery does each thing exactly once.

## Frustration

Once something can be wanted it can be blocked, and blockage is not the
absence of motivation. It is its own state, and it accumulates:

    F[t+1] = rho * F[t] + wanting * (1 - P(success))

The multiplication matters. Failing at something you do not care about
produces nothing; failing repeatedly at something you want badly produces a
great deal. That is the correct ordering and it falls out of the product
rather than needing a rule.

What frustration is *for* is the useful part. Mild frustration should widen
search and prompt a change of strategy, which is the difference between "this
is not working, try another way" and "ERROR". High frustration should trigger
disengagement rather than more of the same, because an agent that responds to
every failure with more effort has no way to ever stop.

That gives an inverted-U in effort against frustration, which is the
Yerkes-Dodson shape: neither zero activation nor runaway activation performs.
The peak is where strategy-switching happens, and both tails are failures of
different kinds.

## Arousal

Arousal tracks the *derivative* of motivation rather than its level, which is
why a cue resolving suddenly produces a jolt and a cue held at high value for
a minute produces nothing.

    A[t+1] = rho * A[t] + rise_in_strongest_motive + |prediction_error|

Aura already owns the autonomic layer this feeds. ``DamasioMarkers`` holds
heart rate, GSR, cortisol and adrenaline; ``AffectiveCircumplex`` holds the
arousal axis that sets sampling temperature. This module computes the conative
contribution and hands it over. It keeps no heart rate of its own, because two
heart rates are two answers to one question.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation

EPS = 1e-12


def half_life_decay(half_life_s: float, elapsed_s: float) -> float:
    """Multiplier for one interval, from a half-life in seconds.

    Stating a half-life rather than a per-tick multiplier is what makes a
    decay rate mean the same thing at any tick rate. A multiplier of 0.9 is
    meaningless until paired with how often it is applied; a half-life of 90
    seconds is 90 seconds however often it is read.
    """
    if half_life_s <= EPS:
        return 0.0
    return 0.5 ** (max(0.0, elapsed_s) / half_life_s)


#: What motivational activation is worth as an estimate of arousal. Middling:
#: it decays on a clock, so it can read high with nothing currently happening.
_CANONICAL_CONATIVE_CONFIDENCE = 0.4


@dataclass
class SatiationState:
    """How thoroughly a motive has been consumed, and how it recovers.

    ``suppression`` is subtracted from the motive's magnitude. It rises on
    consummation and decays back toward zero, so a satisfied motive returns
    rather than being permanently spent.
    """

    key: str
    suppression: float = 0.0
    consummations: int = 0
    last_update: float = field(default_factory=time.time)

    #: How long until half the suppression has lifted. Set to the span of an
    #: exchange rather than of a session: a curiosity satisfied two minutes
    #: ago should have most of its edge back, while one satisfied four seconds
    #: ago should not. Longer would make Aura visit each thing once per hour;
    #: shorter would not suppress at all.
    RECOVERY_HALF_LIFE_S = 120.0

    def decay(self, now: float | None = None) -> float:
        moment = time.time() if now is None else now
        elapsed = max(0.0, moment - self.last_update)
        self.suppression *= half_life_decay(self.RECOVERY_HALF_LIFE_S, elapsed)
        self.last_update = moment
        if self.suppression < 1e-4:
            self.suppression = 0.0
        return self.suppression

    def consume(self, completeness: float) -> float:
        """Record that this motive was satisfied, and by how much.

        ``completeness`` in [0, 1]: a glance at the snail is partial, taking
        it apart is total. Suppression accumulates toward one rather than
        overwriting, so two partial satisfactions add up the way they do.
        """
        self.decay()
        amount = max(0.0, min(1.0, float(completeness)))
        self.suppression = min(1.0, self.suppression + amount * (1.0 - self.suppression))
        self.consummations += 1
        return self.suppression

    def apply(self, magnitude: float) -> float:
        """Attenuate a motive by how recently it was satisfied."""
        return max(0.0, float(magnitude) * (1.0 - self.decay()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "suppression": round(self.suppression, 6),
            "consummations": self.consummations,
        }


@dataclass
class FrustrationState:
    """A wanted thing that keeps not working.

    Frustration is the only state in this package whose *purpose* is to stop
    the behaviour that produced it. Its two thresholds mark the transitions
    that matter: where an agent should change approach, and where it should
    stop.
    """

    key: str
    level: float = 0.0
    attempts: int = 0
    failures: int = 0
    last_update: float = field(default_factory=time.time)

    #: Frustration outlasts the attempt that caused it — that is what makes it
    #: accumulate across tries — but not the session. Five minutes is the span
    #: over which repeated failure at one thing reads as a run rather than as
    #: unrelated events.
    HALF_LIFE_S = 300.0

    #: Above this, keep going but change approach. Reached after roughly three
    #: total failures at something wanted at full strength, which is the point
    #: where a repeated method has been given a fair test.
    SWITCH_THRESHOLD = 0.45
    #: Above this, stop. Reached after roughly seven, where further attempts
    #: with the same resources have stopped being evidence about anything.
    DISENGAGE_THRESHOLD = 0.80

    def decay(self, now: float | None = None) -> float:
        moment = time.time() if now is None else now
        elapsed = max(0.0, moment - self.last_update)
        self.level *= half_life_decay(self.HALF_LIFE_S, elapsed)
        self.last_update = moment
        if self.level < 1e-4:
            self.level = 0.0
        return self.level

    def observe_attempt(self, *, wanting: float, succeeded: bool) -> float:
        """Fold one attempt in. Success discharges; failure accumulates.

        Success does not merely stop the accumulation, it removes what was
        there. An agent that stays frustrated after the thing works has a
        state variable with no way back down.
        """
        self.decay()
        self.attempts += 1
        pull = max(0.0, min(1.0, float(wanting)))
        if succeeded:
            self.level = max(0.0, self.level - pull)
        else:
            self.failures += 1
            self.level = min(1.0, self.level + pull * (1.0 - self.level))
        return self.level

    def should_switch_strategy(self) -> bool:
        return self.SWITCH_THRESHOLD <= self.decay() < self.DISENGAGE_THRESHOLD

    def should_disengage(self) -> bool:
        return self.decay() >= self.DISENGAGE_THRESHOLD

    def effort_multiplier(self) -> float:
        """Effort as a function of frustration: the inverted U.

        Rises to the strategy-switch threshold, where an agent is trying
        hardest and looking for another way, then falls away toward
        disengagement. A monotonic function here would give an agent that
        pushes hardest exactly when it should stop.
        """
        level = self.decay()
        peak = self.SWITCH_THRESHOLD
        if level <= peak:
            return 1.0 + level / max(peak, EPS)
        span = max(1.0 - peak, EPS)
        return max(0.0, 2.0 * (1.0 - (level - peak) / span))

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "level": round(self.level, 6),
            "attempts": self.attempts,
            "failures": self.failures,
            "switch": self.should_switch_strategy(),
            "disengage": self.should_disengage(),
            "effort_multiplier": round(self.effort_multiplier(), 4),
        }


class ConativeDynamics:
    """Satiation, frustration and the arousal a motive contributes."""

    MAX_TRACKED = 256

    def __init__(self) -> None:
        self._satiation: dict[str, SatiationState] = {}
        self._frustration: dict[str, FrustrationState] = {}
        self._arousal: float = 0.0
        self._arousal_updated: float = time.time()
        self._strongest_motive: float = 0.0

    #: Arousal from one event decays to half in this many seconds. Short: an
    #: activation that outlasts its cause by minutes is a mood, and mood is
    #: the affect layer's business rather than this one's.
    AROUSAL_HALF_LIFE_S = 20.0

    # ── satiation ────────────────────────────────────────────────────────

    def satiation(self, key: str) -> SatiationState:
        state = self._satiation.get(key)
        if state is None:
            self._evict(self._satiation)
            state = SatiationState(key=key)
            self._satiation[key] = state
        return state

    def consummate(self, key: str, completeness: float) -> float:
        """Record that a motive was satisfied. Suppresses it, then recovers."""
        return self.satiation(key).consume(completeness)

    def attenuate(self, key: str, magnitude: float) -> float:
        """Apply recent satisfaction to a freshly computed magnitude."""
        state = self._satiation.get(key)
        return magnitude if state is None else state.apply(magnitude)

    # ── frustration ──────────────────────────────────────────────────────

    def frustration(self, key: str) -> FrustrationState:
        state = self._frustration.get(key)
        if state is None:
            self._evict(self._frustration)
            state = FrustrationState(key=key)
            self._frustration[key] = state
        return state

    def observe_attempt(self, key: str, *, wanting: float, succeeded: bool) -> float:
        return self.frustration(key).observe_attempt(
            wanting=wanting, succeeded=succeeded
        )

    def blocked_and_wanted(self) -> list[dict[str, Any]]:
        """Frustrations at or past the strategy-switch threshold."""
        rows = [
            state.to_dict()
            for state in self._frustration.values()
            if state.decay() >= state.SWITCH_THRESHOLD
        ]
        rows.sort(key=lambda row: -row["level"])
        return rows

    # ── arousal ──────────────────────────────────────────────────────────

    def arousal(self) -> float:
        """Current conative contribution to activation, decayed to now."""
        now = time.time()
        elapsed = now - self._arousal_updated
        self._arousal *= half_life_decay(self.AROUSAL_HALF_LIFE_S, elapsed)
        self._arousal_updated = now
        if self._arousal < 1e-4:
            self._arousal = 0.0
        # Motivational activation is one estimator of canonical arousal: how
        # mobilised she is because something matters to her, as against the
        # brainstem reading or the appraisal. Confidence is middling because
        # it decays on a clock and can be high with nothing happening.
        try:
            from core.canonical.state import estimate

            estimate(
                "affect.arousal",
                min(1.0, self._arousal),
                confidence=_CANONICAL_CONATIVE_CONFIDENCE,
                producer="conation",
                note="decayed motive strength",
            )
        except (ImportError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("conation.dynamics", exc, action="canonical estimate skipped")
        return self._arousal

    def register_motive(
        self, strongest_now: float, *, prediction_error: float = 0.0
    ) -> float:
        """Fold this evaluation's strongest motive into activation.

        Only the *rise* counts. A motive that has been at 0.8 for a minute
        contributes nothing further; one that went from 0.12 to 0.91 in a
        single step contributes the difference. That asymmetry is why the
        jolt happens at the smell rather than during the meal.
        """
        current = self.arousal()
        rise = max(0.0, float(strongest_now) - self._strongest_motive)
        self._strongest_motive = max(0.0, min(1.0, float(strongest_now)))
        self._arousal = min(1.0, current + rise + abs(float(prediction_error)))
        self._arousal_updated = time.time()
        return self._arousal

    def couple_to_soma(self) -> dict[str, Any]:
        """Hand the conative arousal term to Aura's existing autonomic layer.

        This module does not hold a heart rate. It reports a contribution and
        the affect engine decides what that does to the body it already owns.
        Returns what was delivered, or the reason nothing was.
        """
        activation = self.arousal()
        if activation <= EPS:
            return {"delivered": False, "reason": "no conative activation"}
        try:
            from core.affect.affective_circumplex import get_circumplex

            circumplex = get_circumplex()
            applied = False
            if circumplex is not None and hasattr(circumplex, "apply_event"):
                # Arousal only. Conation says how activated she is, and says
                # nothing about whether the activation is pleasant — a jolt at
                # a smell and a jolt at a deadline are the same term here.
                circumplex.apply_event(0.0, activation)
                applied = True
            return {
                "delivered": applied,
                "arousal": round(activation, 6),
                "reason": None if applied else "affective circumplex unavailable",
            }
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "conation_dynamics", exc, severity="debug",
                action="conative arousal not delivered to the affect layer",
            )
            return {"delivered": False, "reason": "affect layer unreachable"}

    # ── readout ──────────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        return {
            "arousal": round(self.arousal(), 6),
            "satiated": [
                state.to_dict()
                for state in self._satiation.values()
                if state.decay() > 0.05
            ][:5],
            "frustrated": self.blocked_and_wanted()[:5],
            "tracked": {
                "satiation": len(self._satiation),
                "frustration": len(self._frustration),
            },
        }

    def _evict(self, table: dict[str, Any]) -> None:
        if len(table) < self.MAX_TRACKED:
            return
        stalest = min(table.values(), key=lambda state: state.last_update)
        table.pop(stalest.key, None)
