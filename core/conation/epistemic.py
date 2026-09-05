"""core/conation/epistemic.py — the snail, and why it is not a noisy television.

Curiosity modelled as prediction error is the single most common mistake in
intrinsic motivation, and it fails in a specific, famous way. Point such an
agent at static on a screen and it will watch forever: the pixels are maximally
unpredictable, the error never falls, the reward never stops. Irreducible noise
is indistinguishable from an infinitely deep subject.

Berlyne saw the shape of this in 1954 when he separated perceptual curiosity
from epistemic curiosity, and Oudeyer and Kaplan gave it the fix in 2007:
reward improvement in prediction, not unpredictability. What is interesting is
what you are *currently able to learn something from*. Static teaches nothing,
so its learning progress is zero, so it is boring.

A snail is neither static nor solved. It is the middle case, which is exactly
where a hand goes out.

## The valuation

    C = affordance * [ novelty + information_gain + learning_progress
                       + controllability - irreducible_uncertainty - effort ]

Two things about that expression matter more than its terms.

**The affordance gate is multiplicative.** An action with nothing inspectable
about it scores zero curiosity no matter how novel its surroundings are.
Without the gate, every action in a novel room inherits the room's novelty and
the agent becomes curious about its own idle loop. The gate is what makes
curiosity be *about* something.

**Irreducible uncertainty is subtracted, not ignored.** This is the noisy-TV
guard stated as arithmetic. A source whose unpredictability does not fall with
exposure has its curiosity actively cancelled rather than merely failing to
grow, because a term that only fails to grow still leaves the largest raw
uncertainty winning.

## Weights

The bracketed terms carry equal weight. That is a deliberate refusal to invent
a psychology: nothing has measured, on this system, that information gain
matters more than novelty or less than learning progress, and any other
weighting would assert an ordering no observation here supports. Equal weights
are the maximum-entropy choice given no data. ``core/conation/calibration.py``
holds the hook a learned head would use to earn different ones and reports
``learned=False`` until one does, so a caller can always tell an assumption
from a measurement.

## What is reused

Aura already measures two of these terms and this module does not recompute
them. ``NoveltyMotivation`` in ``core/adaptation/intrinsic_motivation.py``
holds a kernel-density archive of visited states; ``CompetenceRecord`` in the
same module holds the competence derivative, which is learning progress under
its other name. This file reads them. A second novelty estimate would be a
second answer to one question, and CP126 settled which of those a system may
have.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from core.conation.origins import OriginReading, ValueOrigin
from core.runtime.errors import record_degradation

EPS = 1e-12


def categorical_kl(posterior: Sequence[float], prior: Sequence[float]) -> float:
    """D_KL(posterior || prior) in nats, over two categorical distributions.

    Raises on a length mismatch rather than padding, because a prior and a
    posterior of different support are not two beliefs about one question.
    """
    if len(posterior) != len(prior):
        raise ValueError("prior and posterior must share a support")
    total = 0.0
    for p, q in zip(posterior, prior):
        if p > 0.0:
            total += p * math.log((p + EPS) / (q + EPS))
    return max(0.0, total)


def gaussian_entropy_reduction(
    prior_variance: Sequence[float],
    posterior_variance: Sequence[float],
) -> float:
    """Expected information gain for a Gaussian belief, in nats.

    For independent Gaussians the differential entropy is
    ``0.5 * sum(log(2*pi*e*var))``, so the gain from an observation is half
    the summed log ratio of the variances. This is the computable form of
    "how much would doing this change what I know", and it needs only the
    covariance a predictive model already carries.
    """
    if len(prior_variance) != len(posterior_variance):
        raise ValueError("prior and posterior variance must share a dimension")
    total = 0.0
    for before, after in zip(prior_variance, posterior_variance):
        if before <= EPS or after <= EPS:
            continue
        total += 0.5 * math.log(before / after)
    return max(0.0, total)


#: Where the Wundt curve peaks, in normalised arousal potential. The midpoint
#: is the maximum-entropy choice: nothing measured on this system says the
#: sweet spot sits high or low, and putting it anywhere else asserts a taste.
WUNDT_OPTIMUM = 0.5

#: Width of the curve. One quarter of the range puts the half-maximum points
#: at the quartiles, so the middle half of the complexity range scores above
#: half and both extremes fall to about 0.14. That geometry is the claim
#: Wundt's curve makes — interesting things are neither trivial nor
#: impenetrable — and the width follows from it rather than from tuning.
WUNDT_WIDTH = 0.25


def wundt_curve(arousal_potential: float) -> float:
    """Berlyne's inverted U over how much is going on in a stimulus.

    Curiosity does not rise with complexity. It peaks in the middle and falls
    off both sides, and the two tails fail for different reasons: a blank wall
    offers nothing to resolve, and a wall of static offers nothing resolvable.
    A monotonic curiosity function gets the low tail wrong even when an
    irreducible-uncertainty penalty rescues the high one, which is why the
    penalty is not a substitute for this.

    A snail sits near the peak. That is the whole reason a hand goes out.
    """
    potential = max(0.0, min(1.0, float(arousal_potential)))
    z = (potential - WUNDT_OPTIMUM) / WUNDT_WIDTH
    return math.exp(-0.5 * z * z)


def saturate(nats: float) -> float:
    """Map an unbounded information gain into [0, 1).

    ``1 - exp(-x)`` is the natural squashing for a quantity in nats: it is the
    fraction of the belief's uncertainty that the observation would remove, so
    one nat reads 0.63 and four nats read 0.98. A linear cap would make every
    large gain identical, which is wrong in the case that matters — choosing
    between two very informative actions.
    """
    return 1.0 - math.exp(-max(0.0, nats))


@dataclass
class UncertaintyTrace:
    """Per-target history used to tell learnable uncertainty from noise.

    An error that falls with exposure was reducible. An error that stays flat
    across many exposures is the television. The distinction needs history,
    which is the entire reason this class exists rather than a scalar.
    """

    key: str
    errors: list[float] = field(default_factory=list)
    exposures: int = 0

    #: Exposures before the reducible/irreducible split can be called at all.
    #: Three is the smallest count over which a trend has a direction; below
    #: it the trace reports no verdict rather than a weak one.
    MIN_EXPOSURES = 3
    MAX_HISTORY = 64

    def observe(self, error: float) -> None:
        self.errors.append(max(0.0, float(error)))
        if len(self.errors) > self.MAX_HISTORY:
            self.errors.pop(0)
        self.exposures += 1

    def learning_progress(self) -> float | None:
        """Fall in prediction error across the retained window.

        Positive means the target is teaching. ``None`` means too few
        exposures to say, which is different from zero progress.
        """
        if len(self.errors) < self.MIN_EXPOSURES:
            return None
        midpoint = len(self.errors) // 2
        older = self.errors[:midpoint] or self.errors[:1]
        newer = self.errors[midpoint:] or self.errors[-1:]
        return max(0.0, (sum(older) / len(older)) - (sum(newer) / len(newer)))

    def irreducible(self) -> float | None:
        """The floor the error refuses to go below, normalised.

        A target whose error is high and flat has a high floor and is noise. A
        target whose error is falling has a floor below its current error and
        is a subject. ``None`` while support is too thin.
        """
        if len(self.errors) < self.MIN_EXPOSURES:
            return None
        floor = min(self.errors)
        ceiling = max(self.errors)
        if ceiling <= EPS:
            return 0.0
        # A flat trace has floor == ceiling and reads fully irreducible; a
        # trace that fell to nothing reads zero.
        return max(0.0, min(1.0, floor / ceiling)) * max(0.0, min(1.0, floor))

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "exposures": self.exposures,
            "learning_progress": self.learning_progress(),
            "irreducible": self.irreducible(),
        }


class EpistemicValuation:
    """Curiosity that is about something, and that noise cannot capture."""

    MAX_TRACES = 256

    def __init__(self) -> None:
        self._traces: dict[str, UncertaintyTrace] = {}

    # ── reading Aura's existing measurements ─────────────────────────────

    @staticmethod
    def novelty_of(state_vector: Sequence[float] | None) -> tuple[float | None, str]:
        """Novelty from Aura's kernel-density archive, or ``None``.

        Reads ``NoveltyMotivation`` rather than keeping a second archive.
        """
        if state_vector is None:
            return None, "no state vector supplied"
        try:
            import numpy as np

            from core.container import ServiceContainer

            engine = ServiceContainer.get("intrinsic_motivation", default=None)
            novelty_model = getattr(engine, "novelty", None)
            if novelty_model is None or not hasattr(novelty_model, "compute_novelty"):
                return None, "novelty archive unavailable"
            value = float(
                novelty_model.compute_novelty(np.asarray(state_vector, dtype=float))
            )
            if not math.isfinite(value):
                return None, "novelty archive returned a non-finite value"
            return max(0.0, min(1.0, value)), "kernel-density novelty archive"
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "conation_epistemic", exc, severity="debug",
                action="novelty unreadable; term omitted from curiosity",
            )
            return None, "novelty archive unreadable"

    @staticmethod
    def competence_progress(goal_name: str | None) -> tuple[float | None, str]:
        """Learning progress from Aura's competence derivative, or ``None``."""
        if not goal_name:
            return None, "no competence goal named"
        try:
            from core.container import ServiceContainer

            engine = ServiceContainer.get("intrinsic_motivation", default=None)
            competence = getattr(engine, "competence", None)
            records = getattr(competence, "records", None)
            if not isinstance(records, dict):
                return None, "competence records unavailable"
            record = records.get(goal_name)
            if record is None or not hasattr(record, "competence_derivative"):
                return None, f"no competence record for {goal_name}"
            derivative = float(record.competence_derivative())
            if not math.isfinite(derivative):
                return None, "competence derivative non-finite"
            return max(0.0, min(1.0, derivative)), f"competence slope for {goal_name}"
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "conation_epistemic", exc, severity="debug",
                action="competence derivative unreadable; term omitted",
            )
            return None, "competence records unreadable"

    # ── valuation ────────────────────────────────────────────────────────

    def value(
        self,
        key: str,
        *,
        epistemic_affordance: float,
        state_vector: Sequence[float] | None = None,
        competence_goal: str | None = None,
        prior_variance: Sequence[float] | None = None,
        posterior_variance: Sequence[float] | None = None,
        prior_belief: Sequence[float] | None = None,
        posterior_belief: Sequence[float] | None = None,
        controllability: float | None = None,
        arousal_potential: float | None = None,
        instrumental: bool = False,
        irreducible_override: float | None = None,
        effort: float = 0.0,
    ) -> OriginReading:
        """Curiosity about one target.

        ``epistemic_affordance`` is the gate: how much about this target is
        actually inspectable. Zero means there is nothing to find out, and the
        origin reports unavailable — an action is not curious-making merely by
        existing near something novel.

        ``arousal_potential`` is how much is going on in the target, and it
        enters through Wundt's curve rather than linearly. Supplying it is
        what distinguishes the snail from both a blank wall and a wall of
        static.

        ``instrumental`` marks a look taken to get something else.
        Such a look is still worth taking and is still priced here, but it is
        reported as instrumental so it cannot be counted as the autotelic
        case. The distinction is the difference between reading a manual and
        picking up a snail, and only one of those two is what curiosity means
        in the sense this package is about.
        """
        origin = ValueOrigin.EPISTEMIC
        affordance = max(0.0, min(1.0, float(epistemic_affordance)))
        if affordance <= EPS:
            return OriginReading.unavailable(
                origin, "nothing about this target is inspectable"
            )

        terms: dict[str, float] = {}
        sources: list[str] = []

        novelty, novelty_evidence = self.novelty_of(state_vector)
        if novelty is not None:
            terms["novelty"] = novelty
            sources.append(f"novelty {novelty:.3f} ({novelty_evidence})")

        # Expected information gain, from whichever belief shape the caller
        # holds. A Gaussian model carries a covariance; a discrete one carries
        # a distribution. Both answer the same question — how much would doing
        # this change what I know — and supporting only one would make the
        # term unavailable to half the callers that could supply it.
        gain_nats: float | None = None
        if prior_variance is not None and posterior_variance is not None:
            try:
                gain_nats = gaussian_entropy_reduction(prior_variance, posterior_variance)
            except ValueError as exc:
                record_degradation(
                    "conation_epistemic", exc, severity="debug",
                    action="belief variances mismatched; gain term omitted",
                )
        elif prior_belief is not None and posterior_belief is not None:
            try:
                gain_nats = categorical_kl(posterior_belief, prior_belief)
            except ValueError as exc:
                record_degradation(
                    "conation_epistemic", exc, severity="debug",
                    action="belief supports mismatched; gain term omitted",
                )
        if gain_nats is not None:
            terms["information_gain"] = saturate(gain_nats)
            sources.append(f"expected gain {gain_nats:.3f} nats")

        trace = self._traces.get(key)
        progress = trace.learning_progress() if trace is not None else None
        if progress is None:
            progress, progress_evidence = self.competence_progress(competence_goal)
            if progress is not None:
                sources.append(f"learning progress {progress:.3f} ({progress_evidence})")
        else:
            sources.append(
                f"learning progress {progress:.3f} over {trace.exposures} exposures"
            )
        if progress is not None:
            terms["learning_progress"] = progress

        if controllability is not None:
            terms["controllability"] = max(0.0, min(1.0, float(controllability)))
            sources.append(f"controllability {terms['controllability']:.3f}")

        if arousal_potential is not None:
            terms["wundt"] = wundt_curve(arousal_potential)
            sources.append(
                f"Wundt {terms['wundt']:.3f} at potential {float(arousal_potential):.2f}"
            )

        if not terms:
            return OriginReading.unavailable(
                origin, "no epistemic term could be measured"
            )

        # Equal weight over the terms that could actually be measured. A term
        # that had no evidence is absent rather than zero, so its absence does
        # not drag the mean down and pretend to be a measured nothing.
        positive = sum(terms.values()) / len(terms)

        irreducible = trace.irreducible() if trace is not None else None
        if irreducible_override is not None:
            irreducible = max(0.0, min(1.0, float(irreducible_override)))
        penalty = 0.0
        if irreducible is not None:
            penalty += irreducible
            sources.append(f"irreducible uncertainty {irreducible:.3f}")
        penalty += max(0.0, min(1.0, effort))

        magnitude = affordance * max(0.0, positive - penalty)

        return OriginReading(
            origin=origin,
            magnitude=max(0.0, min(1.0, magnitude)),
            available=True,
            evidence="; ".join(sources) if sources else "affordance only",
            detail={
                "affordance": affordance,
                "positive": positive,
                "penalty": penalty,
                "instrumental": 1.0 if instrumental else 0.0,
                **terms,
                **({"irreducible": irreducible} if irreducible is not None else {}),
            },
        )

    # ── learning ─────────────────────────────────────────────────────────

    def observe_error(self, key: str, prediction_error: float) -> UncertaintyTrace:
        """Fold one exposure's prediction error into the target's trace.

        This is what eventually tells a snail from a television. Neither can be
        distinguished on first contact, and a system that claims to know the
        difference before it has looked twice is guessing.
        """
        trace = self._traces.get(key)
        if trace is None:
            if len(self._traces) >= self.MAX_TRACES:
                stalest = min(self._traces.values(), key=lambda t: t.exposures)
                self._traces.pop(stalest.key, None)
            trace = UncertaintyTrace(key=key)
            self._traces[key] = trace
        trace.observe(prediction_error)
        return trace

    def noisy_sources(self) -> list[str]:
        """Targets whose uncertainty has refused to fall. Televisions."""
        out = []
        for trace in self._traces.values():
            irreducible = trace.irreducible()
            progress = trace.learning_progress()
            if irreducible is not None and progress is not None:
                if irreducible > 0.5 and progress <= EPS:
                    out.append(trace.key)
        return out

    def status(self) -> dict[str, Any]:
        noisy = self.noisy_sources()
        return {
            "traces": len(self._traces),
            # Sample and count both. A caller that sees five entries and no
            # count cannot tell five from fifty, and anything it says about
            # how many is then false by construction.
            "noisy_sources": noisy[:5],
            "noisy_source_count": len(noisy),
            "tracked": [trace.to_dict() for trace in list(self._traces.values())[:5]],
        }
