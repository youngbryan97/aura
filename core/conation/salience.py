"""core/conation/salience.py — wanting, liking, and the gap between them.

Berridge and Robinson's incentive-salience account is the reason this file
exists. A cue acquires motivational pull through learning, and that pull is
then re-multiplied at the moment of encounter by the body's current state.
The same cake is a jolt when hungry and scenery when full, and the cached
value of the cake did not change between those two moments. What changed was
the gain.

Zhang, Berridge, Tindell, Smith and Aldridge gave that gain a name and a
computational model in 2009: incentive salience is cached value times a
physiological modulator, not cached value plus one. The multiplicative form is
what produces the observed behaviour, because a deprivation state with nothing
learned to attach to still yields nothing.

    wanting = cached_value ** gamma * exp(eta * kappa)

``kappa`` here is read from Aura's live resource budgets. It is a measured
deficit fraction from ``core/drive_engine.py`` and never a passed-in number,
which is what keeps this an account of her state rather than an account of an
argument somebody supplied.

## Liking is a different predictor

Hedonic impact gets its own model and its own error term. The two are allowed
to disagree, and the disagreement is the useful part: a cue with high cached
pull and repeatedly low experienced liking is a cue whose value has drifted,
and only separate predictors can notice.

Updates are delta-rule. The learning rate is a declared calibration rather
than a number chosen for feel — see ``SalienceCalibration`` below.

## Arousal is the derivative, not the level

The jolt at a smell arrives before any eating, which means it cannot be
tracking the value of the food. It tracks the *change* in pull as the cue
resolves. A cue that has been sitting at high wanting for a minute produces
nothing; a cue that moves wanting from 0.12 to 0.91 in one step produces the
jump.

    activation_delta = max(0, d(wanting)/dt) + abs(prediction_error)

Aura already owns an autonomic layer. ``DamasioMarkers`` in
``core/affect/damasio_v2.py`` holds heart rate, GSR, cortisol and adrenaline,
and ``AffectiveCircumplex`` holds arousal. This module computes the conative
term and hands it to those, and it does not keep a heart rate of its own.
Two heart rates would be two answers to one question, which CP126 settled.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation

#: Guard against a zero-probability term in a log or a divide.
EPS = 1e-12


@dataclass(frozen=True, slots=True)
class SalienceCalibration:
    """Parameters of the salience model, with their defaults justified.

    Every value here is a real parameter of the Berridge form, and none of
    them has been measured on this system yet. The defaults are chosen to be
    the *assumption-free* setting rather than a tuned one, so that the model
    asserts as little as possible until evidence arrives:

    ``gamma = 1.0`` makes wanting linear in cached value. Any other exponent
    claims a curvature — diminishing or accelerating returns on learned value
    — that nothing here has observed.

    ``eta = 1.0`` makes a fully depleted budget multiply pull by ``e``, which
    is the unit-gain reading of the exponential term. It says deprivation
    matters at the natural scale of the function, and nothing more specific.

    ``liking_learning_rate = 0.1`` gives an effective averaging window near
    ten contacts (a delta rule with rate a has time constant 1/a). Ten is the
    smallest window over which a hedonic estimate stops tracking single
    outcomes, which is the property wanted; it is a stated assumption, not a
    measurement.

    ``wanting_learning_rate = 0.1`` matches it, so that neither predictor
    outruns the other by construction and any divergence between wanting and
    liking is a fact about the incentive rather than about the rates.

    ``core/conation/calibration.py`` holds the hook a learned head would use
    to replace them and records the hedonic prediction errors that would grade
    it. No head holds authority today, so the defaults stand and the readout
    says so rather than presenting them as measurements.
    """

    gamma: float = 1.0
    eta: float = 1.0
    liking_learning_rate: float = 0.1
    wanting_learning_rate: float = 0.1

    #: Samples of contact before a hedonic estimate is reported as usable at
    #: all. Below this the predictor still updates, but reports ``None`` so a
    #: caller cannot mistake one taste for a preference. Three is the smallest
    #: count at which a mean has any dispersion behind it.
    min_liking_samples: int = 3

    #: Whether a learned head currently supplies these values. False means the
    #: documented defaults above are in force.
    learned: bool = False
    source: str = "declared_default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "gamma": self.gamma,
            "eta": self.eta,
            "liking_learning_rate": self.liking_learning_rate,
            "wanting_learning_rate": self.wanting_learning_rate,
            "min_liking_samples": self.min_liking_samples,
            "learned": self.learned,
            "source": self.source,
        }


@dataclass
class CueRecord:
    """Learned state for one incentive: what it pulls, and what it delivers."""

    key: str
    cached_value: float = 0.0
    contacts: int = 0

    #: Hedonic estimate and its support count. Held apart from cached_value on
    #: purpose; these are the two numbers allowed to disagree.
    liking_estimate: float = 0.0
    liking_samples: int = 0

    #: Recent hedonic prediction errors, for reporting drift.
    liking_errors: deque[float] = field(default_factory=lambda: deque(maxlen=32))
    #: Contacts where pull and hedonic impact pointed opposite ways.
    dissociations: int = 0

    last_wanting: float = 0.0
    last_seen: float = field(default_factory=time.time)

    #: Which origins have ever supplied this cue's value, and how often. A
    #: cached value is learned pull whose source is historical, and without
    #: this the history is lost: after enough contacts a borrowed want and a
    #: bodily one are the same number. Keeping the tally is the same refusal
    #: the vicarious ledger makes, applied to the cache that outlives it.
    origin_counts: dict[str, int] = field(default_factory=dict)

    def attribute(self, origin: str) -> None:
        self.origin_counts[origin] = self.origin_counts.get(origin, 0) + 1

    def provenance(self) -> str | None:
        """The origin that has most often supplied this cue's value."""
        if not self.origin_counts:
            return None
        return max(self.origin_counts.items(), key=lambda kv: kv[1])[0]

    def liking_prediction(self, min_samples: int) -> float | None:
        """Hedonic estimate, or ``None`` while support is too thin to mean it."""
        if self.liking_samples < min_samples:
            return None
        return self.liking_estimate

    def drift(self) -> float:
        """Mean signed hedonic error over the retained window.

        Negative means this cue has been consistently overvalued: it keeps
        pulling harder than it pays. That is the measurable signature of a
        wanting/liking divergence, and it is what a self-model can act on.
        """
        if not self.liking_errors:
            return 0.0
        return sum(self.liking_errors) / len(self.liking_errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "cached_value": round(self.cached_value, 6),
            "contacts": self.contacts,
            "liking_estimate": round(self.liking_estimate, 6),
            "liking_samples": self.liking_samples,
            "dissociations": self.dissociations,
            "drift": round(self.drift(), 6),
            "last_wanting": round(self.last_wanting, 6),
            "provenance": self.provenance(),
            "origin_counts": dict(self.origin_counts),
        }


class IncentiveSalience:
    """The wanting/liking pair, with physiological gain read from live state.

    Holds one ``CueRecord`` per incentive key. Bounded: the least recently
    seen record is evicted once the table is full, because an unbounded table
    keyed on arbitrary strings is a memory leak with a plausible name.
    """

    #: Records retained. Sized to hold a long conversation's worth of distinct
    #: incentives without becoming a store; the ontogenetic corpus is where
    #: durable history belongs.
    MAX_RECORDS = 512

    def __init__(self, calibration: SalienceCalibration | None = None) -> None:
        self._calibration = calibration or SalienceCalibration()
        self._records: dict[str, CueRecord] = {}
        #: Wanting at the previous evaluation, per key, for the derivative.
        self._previous: dict[str, tuple[float, float]] = {}

    @property
    def calibration(self) -> SalienceCalibration:
        return self._calibration

    def adopt_calibration(self, calibration: SalienceCalibration) -> None:
        """Install a calibration, normally from a promoted ontogenetic head."""
        self._calibration = calibration

    # ── measurement ──────────────────────────────────────────────────────

    def deprivation_gain(self, budget_name: str | None) -> tuple[float, str]:
        """Read kappa: the live deficit fraction of a named resource budget.

        Returns the fraction and an evidence string naming the measurement.
        A budget that does not exist yields zero gain and says so, because a
        homeostatic claim about a budget Aura does not have is not a small
        error, it is a fabricated one.
        """
        if not budget_name:
            return 0.0, "no homeostatic target named"
        try:
            from core.container import ServiceContainer

            engine = ServiceContainer.get("drive_engine", default=None)
            if engine is None:
                engine = ServiceContainer.get("motivation_engine", default=None)
            budgets = getattr(engine, "budgets", None)
            if not isinstance(budgets, dict):
                return 0.0, "drive engine unavailable"
            budget = budgets.get(budget_name)
            if budget is None:
                return 0.0, f"no budget named {budget_name}"
            capacity = float(getattr(budget, "capacity", 0.0) or 0.0)
            level = float(getattr(budget, "level", 0.0) or 0.0)
            if capacity <= EPS:
                return 0.0, f"budget {budget_name} has no capacity"
            deficit = max(0.0, min(1.0, 1.0 - level / capacity))
            return deficit, f"{budget_name} budget at {level:.1f} of {capacity:.1f}"
        except (ImportError, AttributeError, TypeError, ValueError, KeyError) as exc:
            record_degradation(
                "conation_salience", exc, severity="debug",
                action="deprivation gain unreadable; homeostatic origin withheld",
            )
            return 0.0, "drive engine unreadable"

    def wanting(
        self,
        key: str,
        *,
        cached_value: float | None,
        deprivation: float,
        relevance: float = 1.0,
        cue_salience: float = 0.0,
    ) -> float:
        """Incentive salience for one cue under the current deprivation.

        ``cached_value`` of ``None`` falls back to the learned record, and
        with no record either the learned term is zero. Perceptual salience
        adds on top, because an unlearned cue can still pull by standing out —
        that is what makes a first encounter possible at all.
        """
        cal = self._calibration
        record = self._records.get(key)
        learned = cached_value
        if learned is None:
            learned = record.cached_value if record is not None else 0.0
        learned = max(0.0, min(1.0, float(learned)))
        kappa = max(0.0, min(1.0, deprivation)) * max(0.0, min(1.0, relevance))

        gain = math.exp(cal.eta * kappa)
        pull = (learned ** cal.gamma) * gain
        pull += max(0.0, min(1.0, cue_salience))
        # Bounded by the largest value the learned term can reach, so that a
        # cue at full learned value under full deprivation with full
        # perceptual salience reads 1.0 rather than an unbounded number that
        # no downstream weight can be reasoned about.
        ceiling = math.exp(cal.eta) + 1.0
        return max(0.0, min(1.0, pull / ceiling))

    def activation_delta(self, key: str, wanting_now: float) -> tuple[float, float]:
        """The conative term for arousal: positive change in pull per second.

        Returns ``(delta, rate)``. The jolt is the rate at which wanting rose,
        which is why a cue held at high value for a minute produces nothing
        and a cue that resolves suddenly produces the jump.
        """
        now = time.time()
        previous = self._previous.get(key)
        self._previous[key] = (wanting_now, now)
        if previous is None:
            # A first sighting is a rise from nothing, over the shortest
            # interval that can be measured rather than over zero.
            return max(0.0, wanting_now), max(0.0, wanting_now)
        prior_value, prior_time = previous
        elapsed = max(now - prior_time, 1e-3)
        rise = wanting_now - prior_value
        return max(0.0, rise), max(0.0, rise) / elapsed

    def predicted_liking(self, key: str) -> float | None:
        """Hedonic prediction for a cue, or ``None`` with too little contact."""
        record = self._records.get(key)
        if record is None:
            return None
        return record.liking_prediction(self._calibration.min_liking_samples)

    # ── learning ─────────────────────────────────────────────────────────

    def observe(self, key: str, wanting_now: float) -> CueRecord:
        """Note that this cue was evaluated, holding the record current."""
        record = self._touch(key)
        record.last_wanting = wanting_now
        record.last_seen = time.time()
        return record

    def attribute(self, key: str, origin: Any) -> None:
        """Record which origin supplied this cue's value on this evaluation."""
        self._touch(key).attribute(str(origin))

    def provenance(self, key: str) -> str | None:
        """Where this cue's cached pull came from, historically."""
        record = self._records.get(key)
        return None if record is None else record.provenance()

    def record_outcome(
        self,
        key: str,
        *,
        experienced_liking: float,
        realised_pull: float | None = None,
    ) -> dict[str, Any]:
        """Fold one contact into both predictors, keeping the errors apart.

        ``experienced_liking`` is the hedonic impact of contact, in [-1, 1].
        ``realised_pull`` is how much pull the cue turned out to exert; with
        none supplied, the last evaluated wanting stands in.

        Returns both errors and whether they dissociated. The caller decides
        what a dissociation means; this layer only refuses to hide it.
        """
        cal = self._calibration
        record = self._touch(key)
        liking = max(-1.0, min(1.0, float(experienced_liking)))
        pull = record.last_wanting if realised_pull is None else float(realised_pull)
        pull = max(0.0, min(1.0, pull))

        prior_liking = record.liking_prediction(cal.min_liking_samples)
        liking_error = None if prior_liking is None else liking - prior_liking

        record.liking_estimate += cal.liking_learning_rate * (liking - record.liking_estimate)
        record.liking_samples += 1
        record.contacts += 1
        if liking_error is not None:
            record.liking_errors.append(liking_error)

        # Cached value tracks realised pull, which is what makes a cue that
        # keeps delivering nothing eventually stop pulling — slowly, which is
        # also what the phenomenon does.
        record.cached_value += cal.wanting_learning_rate * (pull - record.cached_value)
        record.cached_value = max(0.0, min(1.0, record.cached_value))

        dissociated = (pull >= 0.5 and liking <= 0.0) or (pull <= 0.2 and liking >= 0.5)
        if dissociated:
            record.dissociations += 1

        return {
            "key": key,
            "epsilon_liking": liking_error,
            "epsilon_wanting": pull - record.cached_value,
            "dissociated": dissociated,
            "drift": record.drift(),
            "contacts": record.contacts,
        }

    # ── readout ──────────────────────────────────────────────────────────

    def overvalued(self, *, min_contacts: int = 3) -> list[tuple[str, float]]:
        """Cues that pull harder than they pay, worst first.

        This is the list a self-model reads to answer "what do I keep
        reaching for that does not deliver". It exists because two predictors
        make the question answerable; one predictor makes it unaskable.
        """
        out = [
            (record.key, record.drift())
            for record in self._records.values()
            if record.contacts >= min_contacts and record.drift() < 0.0
        ]
        out.sort(key=lambda pair: pair[1])
        return out

    def status(self) -> dict[str, Any]:
        dissociations = sum(r.dissociations for r in self._records.values())
        contacts = sum(r.contacts for r in self._records.values())
        return {
            "records": len(self._records),
            "contacts": contacts,
            "dissociations": dissociations,
            "dissociation_rate": (dissociations / contacts) if contacts else 0.0,
            "overvalued": [key for key, _ in self.overvalued()[:5]],
            "calibration": self._calibration.to_dict(),
        }

    # ── internals ────────────────────────────────────────────────────────

    def _touch(self, key: str) -> CueRecord:
        record = self._records.get(key)
        if record is None:
            self._evict_if_full()
            record = CueRecord(key=key)
            self._records[key] = record
        return record

    def _evict_if_full(self) -> None:
        if len(self._records) < self.MAX_RECORDS:
            return
        oldest = min(self._records.values(), key=lambda r: r.last_seen)
        self._records.pop(oldest.key, None)
        self._previous.pop(oldest.key, None)
