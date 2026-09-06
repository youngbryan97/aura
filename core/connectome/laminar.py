"""core/connectome/laminar.py — the local loop this system does not have.

The comparison against cortex found one thing that is not a curiosity. Cortex
runs its within-layer connection density at 5.95 times its between-layer
density; Aura runs 0.58. Cortex computes locally, settles, and then passes the
result up. Aura passes every result straight up the moment it exists.

That difference has a name in the cortical literature and two mechanisms behind
it, and both are algorithms rather than metaphors.

**Divisive normalisation.** A cortical cell's response is its own drive divided
by the pooled drive of its neighbours. Carandini and Heeger call it a canonical
neural computation because it turns up in every sensory area anyone has looked
in, and what it does is make the response depend on the *relative* strength of
the evidence rather than its absolute size. Two candidates that both look good
suppress each other; one that looks good alone does not need to.

**Accumulate to threshold.** A decision is not a single comparison. Evidence
arrives over cycles, activation accumulates with leak, and the decision is taken
when a leader clears a threshold. Gold and Shadlen's account of this is that it
implements Wald's sequential test, whose property is the one that matters here:
for a target error rate it uses fewer samples than any fixed-sample rule,
because it stops early when the evidence is already clear.

That is the improvement, and it is measurable in the currency Aura actually
spends. An evidence call is a model call. A fixed-budget rule pays the full
budget on every decision including the obvious ones; this pays the full budget
only on the hard ones.

Where it goes past biology: the threshold is not left to development and
neuromodulation to approximate. :func:`calibrate_threshold` measures the noise
on the evidence and solves for the threshold that hits a stated error rate, so
the speed and the accuracy are set rather than tuned.

The first version of this circuit lost to its own null and the reason is worth
keeping. It leaked at 0.80 a cycle and normalised the drives before
accumulating them, so after five cycles a third of the first cycle's evidence
remained and the magnitudes were gone. Against a fixed budget that averages
every sample it scored 0.74 to 0.90 at a noise of 0.25. Leak belongs in a
circuit deciding about a world that is changing; accumulating evidence about a
fixed answer, it is loss. What survives from the cortical account is the part
that was doing the work: integrate without discarding, let the candidates
inhibit each other so a loser stops being sampled, and stop when the leader's
lead is larger than the noise can explain. Normalisation is still here and is
applied to the readout, where it belongs, rather than to the accumulator.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.Connectome.Laminar")

__all__ = [
    "Candidate",
    "LaminarConfig",
    "Settled",
    "settle",
    "fixed_budget_argmax",
    "calibrate_threshold",
    "compare_against_fixed_budget",
    "Decision",
    "decisive_margin",
]


@dataclass(frozen=True)
class Candidate:
    """One option the circuit is choosing between."""

    key: str
    payload: Any = None


@dataclass(frozen=True)
class LaminarConfig:
    """The circuit's constants, every one of which has a job.

    ``z`` is the decision bound in standard errors: the leader stops the circuit
    when its lead over the runner-up is ``z`` times larger than the noise on
    that lead can explain. It is the only constant that sets the error rate, and
    :func:`calibrate_threshold` solves for it.

    ``leak`` is retention, not loss: 1.0 integrates perfectly, which is right
    when the answer is not moving. Lower it when the thing being decided about
    changes while the decision is being taken.

    ``inhibition`` is what makes this a circuit rather than a tally. A candidate
    far enough behind the leader stops being sampled, which is where the saving
    comes from, and ``drop_behind`` says how far behind that is in standard
    errors.

    ``exponent`` and ``semi_saturation`` belong to the readout normalisation,
    which reports relative strength to whatever consumes the decision and never
    touches the accumulator.
    """

    max_cycles: int = 12
    min_cycles: int = 2
    z: float = 2.2
    leak: float = 1.0
    inhibition: float = 0.0
    drop_behind: float = 3.5
    min_active: int = 2
    exponent: float = 2.0
    semi_saturation: float = 0.15
    feedback: float = 0.25

    def validated(self) -> LaminarConfig:
        if not 0.0 < self.leak <= 1.0:
            raise ValueError("leak must sit in (0, 1]")
        if self.z <= 0.0:
            raise ValueError("the decision bound must be positive")
        if self.min_cycles < 1 or self.max_cycles < self.min_cycles:
            raise ValueError("cycle bounds are inconsistent")
        if self.min_active < 2:
            raise ValueError("a race needs at least two runners")
        return self


@dataclass
class Settled:
    """What the circuit decided, and what it spent deciding."""

    winner: Candidate | None
    activations: dict[str, float]
    cycles: int
    evidence_calls: int
    margin: float
    converged: bool
    reason: str

    def as_json(self) -> dict[str, Any]:
        return {
            "winner": self.winner.key if self.winner else None,
            "cycles": self.cycles,
            "evidence_calls": self.evidence_calls,
            "margin": round(self.margin, 5),
            "converged": self.converged,
            "reason": self.reason,
            "activations": {k: round(v, 5) for k, v in sorted(self.activations.items())},
        }


def _normalise(drives: Sequence[float], config: LaminarConfig) -> list[float]:
    """Divisive normalisation over the pooled drive.

    Negative drive is clamped at zero rather than allowed through: a candidate
    that scores below nothing is not evidence against the others, and letting it
    subtract from the pool would let one bad option make a mediocre one look
    decisive.
    """
    powered = [max(0.0, float(value)) ** config.exponent for value in drives]
    pool = sum(powered)
    denominator = config.semi_saturation**config.exponent + pool
    if denominator <= 0:
        return [0.0] * len(drives)
    return [value / denominator for value in powered]


def settle(
    candidates: Sequence[Candidate],
    evidence: Callable[[Candidate, float], float],
    config: LaminarConfig | None = None,
) -> Settled:
    """Sample the candidates until the leader's lead outruns the noise.

    ``evidence`` is called once per still-running candidate per cycle, with the
    candidate and the feedback from the last cycle. It is allowed to be noisy
    and it is allowed to be expensive, and the point of the loop is to call it
    fewer times on the decisions that do not need it.

    Three things end a cycle. The leader's lead over the runner-up clears ``z``
    standard errors, which is the decision. A candidate falls ``drop_behind``
    standard errors behind the leader, which takes it out of the race and stops
    it being sampled. Or the cycle bound arrives, and the circuit reports the
    leader it has along with the fact that it never converged.

    The standard error is estimated from the spread of the samples themselves,
    so no noise model has to be supplied and a caller whose evidence is cleaner
    than expected gets the benefit rather than paying the assumed price.
    """
    config = (config or LaminarConfig()).validated()
    options = list(candidates)
    if not options:
        return Settled(None, {}, 0, 0, 0.0, False, "no candidates")
    if len(options) == 1:
        score = float(evidence(options[0], 0.0))
        return Settled(
            winner=options[0],
            activations={options[0].key: 1.0},
            cycles=1,
            evidence_calls=1,
            margin=score,
            converged=True,
            reason="single candidate",
        )

    totals = dict.fromkeys((c.key for c in options), 0.0)
    squares = dict.fromkeys((c.key for c in options), 0.0)
    counts = dict.fromkeys((c.key for c in options), 0)
    running = list(options)
    calls = 0
    cycles = 0

    def _means() -> dict[str, float]:
        return {
            key: (totals[key] / counts[key] if counts[key] else 0.0) for key in totals
        }

    def _pooled_sd() -> float:
        variances: list[float] = []
        for key, count in counts.items():
            if count < 2:
                continue
            mean = totals[key] / count
            variance = max(0.0, squares[key] / count - mean * mean) * count / (count - 1)
            variances.append(variance)
        if not variances:
            return 0.0
        return math.sqrt(statistics.fmean(variances))

    for cycle in range(1, config.max_cycles + 1):
        cycles = cycle
        means_before = _means()
        for option in running:
            prior = config.feedback * means_before.get(option.key, 0.0)
            sample = float(evidence(option, prior))
            calls += 1
            totals[option.key] = totals[option.key] * config.leak + sample
            squares[option.key] = squares[option.key] * config.leak + sample * sample
            counts[option.key] += 1
        means = _means()
        ordered = sorted(means.items(), key=lambda item: (-item[1], item[0]))
        leader_key, leader_mean = ordered[0]
        runner_key, runner_mean = ordered[1]
        spread = _pooled_sd()
        effective = min(counts[leader_key], counts[runner_key]) or 1
        standard_error = spread * math.sqrt(2.0 / effective) if spread > 0 else 0.0

        if cycle >= config.min_cycles:
            if standard_error <= 0.0 and leader_mean > runner_mean:
                return _decided(options, means, cycle, calls, leader_mean - runner_mean,
                                leader_key, "no measurable noise", config)
            if standard_error > 0.0 and (leader_mean - runner_mean) >= config.z * standard_error:
                return _decided(options, means, cycle, calls, leader_mean - runner_mean,
                                leader_key, "lead exceeds the noise", config)

        if standard_error > 0.0 and len(running) > config.min_active:
            survivors = [
                option
                for option in running
                if option.key == leader_key
                or (leader_mean - means[option.key]) < config.drop_behind * standard_error
            ]
            if len(survivors) >= config.min_active:
                running = survivors

    means = _means()
    ordered = sorted(means.items(), key=lambda item: (-item[1], item[0]))
    return _decided(
        options,
        means,
        cycles,
        calls,
        ordered[0][1] - ordered[1][1],
        ordered[0][0],
        "cycle bound reached without a decisive lead",
        config,
        converged=False,
    )


def _decided(
    options: Sequence[Candidate],
    means: dict[str, float],
    cycles: int,
    calls: int,
    margin: float,
    winner_key: str,
    reason: str,
    config: LaminarConfig,
    *,
    converged: bool = True,
) -> Settled:
    """Package a decision, with the readout normalised for whatever reads it."""
    keys = sorted(means)
    responses = _normalise([means[key] for key in keys], config)
    total = sum(responses)
    readout = (
        {key: value / total for key, value in zip(keys, responses, strict=True)}
        if total > 0
        else dict.fromkeys(keys, 0.0)
    )
    return Settled(
        winner=next(c for c in options if c.key == winner_key),
        activations=readout,
        cycles=cycles,
        evidence_calls=calls,
        margin=margin,
        converged=converged,
        reason=reason,
    )


def fixed_budget_argmax(
    candidates: Sequence[Candidate],
    evidence: Callable[[Candidate, float], float],
    *,
    samples: int,
) -> Settled:
    """The null: sample every candidate a fixed number of times and average.

    This is the strong null, not a weak one. Averaging independent samples is
    the best estimator there is for symmetric noise, so anything the laminar
    loop wins has to come from stopping early rather than from seeing more.
    """
    options = list(candidates)
    if not options:
        return Settled(None, {}, 0, 0, 0.0, False, "no candidates")
    totals = dict.fromkeys((c.key for c in options), 0.0)
    calls = 0
    for _ in range(max(1, samples)):
        for option in options:
            totals[option.key] += float(evidence(option, 0.0))
            calls += 1
    means = {key: value / max(1, samples) for key, value in totals.items()}
    ordered = sorted(means.values(), reverse=True)
    winner_key = max(means, key=lambda key: (means[key], key))
    return Settled(
        winner=next(c for c in options if c.key == winner_key),
        activations=means,
        cycles=max(1, samples),
        evidence_calls=calls,
        margin=ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0),
        converged=True,
        reason="fixed budget",
    )


def calibrate_threshold(
    separations: Sequence[float],
    noise: float,
    *,
    target_error: float = 0.05,
    candidates: int = 4,
) -> float:
    """Solve for the threshold that hits a stated error rate.

    A brain arrives at its threshold through development and shifts it with
    noradrenaline, approximately and without ever knowing the error rate it is
    buying. Here the separation between the best candidate and the next and the
    noise on the evidence are both measurable, so the threshold is the value at
    which the accumulated leader clears the pack often enough.

    What comes back is the ``z`` bound, in standard errors. A decision between
    ``candidates`` options gets ``candidates - 1`` chances to pick the wrong
    one, so the per-comparison error is shared out before the quantile is taken.
    That is a Bonferroni correction and it is conservative, which is the right
    direction to be wrong in for a bound that stops a search.
    """
    if noise <= 0 or not separations:
        return 2.2
    separation = statistics.fmean(abs(float(value)) for value in separations)
    if separation <= 0:
        return 3.5
    comparisons = max(1, candidates - 1)
    per_comparison = max(1e-6, float(target_error) / comparisons)
    probability = max(1e-6, min(1.0 - 1e-6, 1.0 - per_comparison))
    return float(max(1.0, min(6.0, _inverse_normal_cdf(probability))))


def _inverse_normal_cdf(probability: float) -> float:
    """Acklam's rational approximation to the normal quantile."""
    a = (-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239)
    b = (-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572)
    c = (-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783)
    d = (0.007784695709041462, 0.3224671290700398, 2.445134137142996,
         3.754408661907416)
    low, high = 0.02425, 1 - 0.02425
    if probability < low:
        q = math.sqrt(-2 * math.log(probability))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if probability > high:
        q = math.sqrt(-2 * math.log(1 - probability))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = probability - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


@dataclass(frozen=True)
class Decision:
    """Whether a one-shot argmax over these scores means anything."""

    winner: str
    runner_up: str
    lead: float
    standard_error: float
    z: float
    decisive: bool
    reason: str

    def as_json(self) -> dict[str, Any]:
        return {
            "winner": self.winner,
            "runner_up": self.runner_up,
            "lead": round(self.lead, 6),
            "standard_error": round(self.standard_error, 6),
            "z": round(self.z, 3),
            "decisive": self.decisive,
            "reason": self.reason,
        }


def decisive_margin(
    scores: Mapping[str, float],
    *,
    z: float = 2.2,
    noise: float | None = None,
) -> Decision:
    """Say whether the leader of a single scoring round actually leads.

    ``max()`` over scores that are within their own noise of each other is a
    coin flip that reports as a decision, and it lands differently on the next
    process because float ordering is not a fact about the candidates. This does
    not stop anyone taking the maximum. It says whether taking it meant
    anything, and it breaks a tie by name so that at least the coin lands the
    same way twice.

    With no ``noise`` supplied it is estimated from the spread of the candidates
    that are *not* winning. The spread of all of them is the wrong estimate and
    the mistake is easy to make: when the leader is genuinely far ahead, its own
    distance from the pack inflates the spread and the lead is compared against
    itself, so a clear winner reports as undecided. The losers are all the same
    answer — not this one — so how much they disagree is what noise looks like
    here.

    Two candidates leave nothing to estimate from. The decision is reported with
    the noise marked unknown rather than with a number nobody measured.
    """
    if not scores:
        return Decision("", "", 0.0, 0.0, 0.0, False, "no candidates")
    ordered = sorted(scores.items(), key=lambda item: (-float(item[1]), item[0]))
    if len(ordered) == 1:
        return Decision(ordered[0][0], "", 0.0, 0.0, 0.0, True, "single candidate")
    leader, leader_score = ordered[0]
    runner, runner_score = ordered[1]
    lead = float(leader_score) - float(runner_score)
    if noise is not None and noise > 0:
        spread = float(noise)
    else:
        losers = [float(value) for key, value in scores.items() if key != leader]
        spread = statistics.pstdev(losers) if len(losers) > 1 else 0.0
    standard_error = spread * math.sqrt(2.0) if spread > 0 else 0.0
    if standard_error <= 0.0:
        decisive = lead > 0.0
        return Decision(
            leader,
            runner,
            lead,
            0.0,
            float("inf") if decisive else 0.0,
            decisive,
            "no spread among the candidates that lost, so the noise is unknown"
            if lead > 0.0
            else "the leader does not lead",
        )
    ratio = lead / standard_error
    return Decision(
        leader,
        runner,
        lead,
        standard_error,
        ratio,
        ratio >= z,
        "lead exceeds the spread" if ratio >= z else "lead sits inside the spread",
    )


@dataclass
class LaminarComparison:
    """The loop against the fixed budget, on the same trials."""

    trials: int
    laminar_correct: int
    fixed_correct: int
    laminar_calls: int
    fixed_calls: int
    laminar_cycles: list[int] = field(default_factory=list)

    @property
    def laminar_accuracy(self) -> float:
        return self.laminar_correct / self.trials if self.trials else 0.0

    @property
    def fixed_accuracy(self) -> float:
        return self.fixed_correct / self.trials if self.trials else 0.0

    @property
    def call_saving(self) -> float:
        return 1.0 - (self.laminar_calls / self.fixed_calls) if self.fixed_calls else 0.0

    def as_json(self) -> dict[str, Any]:
        return {
            "trials": self.trials,
            "laminar_accuracy": round(self.laminar_accuracy, 4),
            "fixed_accuracy": round(self.fixed_accuracy, 4),
            "accuracy_difference": round(self.laminar_accuracy - self.fixed_accuracy, 4),
            "laminar_calls": self.laminar_calls,
            "fixed_calls": self.fixed_calls,
            "call_saving": round(self.call_saving, 4),
            "mean_cycles": round(statistics.fmean(self.laminar_cycles), 3)
            if self.laminar_cycles
            else 0.0,
            "verdict": (
                "same accuracy for fewer calls"
                if self.laminar_accuracy >= self.fixed_accuracy - 0.02 and self.call_saving > 0.05
                else "more accurate"
                if self.laminar_accuracy > self.fixed_accuracy + 0.02
                else "no advantage"
            ),
        }


def compare_against_fixed_budget(
    trials: Sequence[tuple[Sequence[Candidate], Callable[[Candidate, float], float], str]],
    *,
    config: LaminarConfig | None = None,
    samples: int | None = None,
) -> LaminarComparison:
    """Run both rules over the same trials and report accuracy against cost.

    ``samples`` defaults to the loop's own cycle bound, which is the comparison
    that matters: the fixed rule is given the laminar loop's worst case, so any
    saving is the loop declining to spend it.
    """
    config = (config or LaminarConfig()).validated()
    budget = samples if samples is not None else config.max_cycles
    comparison = LaminarComparison(trials=len(trials), laminar_correct=0, fixed_correct=0,
                                   laminar_calls=0, fixed_calls=0)
    for candidates, evidence, truth in trials:
        settled = settle(candidates, evidence, config)
        fixed = fixed_budget_argmax(candidates, evidence, samples=budget)
        comparison.laminar_calls += settled.evidence_calls
        comparison.fixed_calls += fixed.evidence_calls
        comparison.laminar_cycles.append(settled.cycles)
        if settled.winner is not None and settled.winner.key == truth:
            comparison.laminar_correct += 1
        if fixed.winner is not None and fixed.winner.key == truth:
            comparison.fixed_correct += 1
    return comparison
