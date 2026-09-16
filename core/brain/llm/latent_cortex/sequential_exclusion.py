"""Sampling without replacement in answer space.

THE ARGUMENT
────────────
Every RLC attempt so far has tried to make a fixed checkpoint compute
something better. That is the hard version of the problem, and the honest
objection stands: nothing guarantees a frozen checkpoint internalises a broad
gain without empirical evidence.

So do not ask it to. Ask instead what property of the SEARCH can be improved
without touching a weight — because there the guarantee is arithmetic.

Let the model's answer distribution over a finite answer set A be p, and let
A* ⊆ A be the correct answers with total mass p*.

  i.i.d. best-of-N with a sound verifier:
      P(success) = 1 − (1 − p*)^N

  sequential exclusion — after k refutations covering incorrect mass m_k,
  draw k+1 comes from p restricted to A \\ R_k and renormalised:
      P(draw k+1 correct) = p* / (1 − m_k)   ≥   p*
      P(success in N)     = 1 − Π_{k<N} (1 − p*/(1 − m_k))

Every factor is no larger than the i.i.d. one and strictly smaller as soon as
m_k > 0. So exclusion dominates i.i.d. for every N, every p, every p*. That
is not a hypothesis about a model; it is a statement about renormalising a
measure after removing mass.

WHY IT SHOULD BE LARGE HERE SPECIFICALLY
────────────────────────────────────────
The dominance is strict in proportion to how PEAKED p is — and this system
has measured its own peakedness twice, in two different ways, without
naming it as such:

  cos(pass1, pass2) = 0.9994  the recurrence had nothing to disagree with;
  "collapse is cheapest"      branches falling into one local basin.

Both say the same thing: the sampler keeps redrawing the same answer. Under
i.i.d. sampling, N draws from a distribution with a 0.7-mass mode spend ~70%
of the budget re-deriving one answer. Best-of-8 is then best-of-2 or 3 in
anything that matters, which is a complete explanation for why more branches
and more depth have bought so little.

Exclusion converts the peak from a liability into an asset. The bigger the
mode, the more mass one refutation removes, and the more the remaining mass
redistributes toward everything else — including the correct answer.

Worked example, p(wrong mode) = 0.70, p* = 0.05, N = 8:
    i.i.d.      1 − 0.95^8                        = 33.7%
    exclusion   mode refuted on draw 1, p* now
                0.05/0.30 = 16.7% per draw        ≈ 76%
Both calculations use the same model, checkpoint and eight forward passes.

THE THREE WAYS IT FAILS, ALL MEASURED HERE
──────────────────────────────────────────
The theorem has three premises, and this module instruments all of them, so
a null result is DIAGNOSTIC rather than mysterious:

  soundness   the verifier must only refute incorrect answers. A refuted
              correct answer excludes the truth permanently. Tracked as
              ``gold_exclusions`` — one is a defect, not noise;
  compliance  the model must actually avoid what was excluded. If it
              redraws the excluded answer anyway, exclusion is nominal and
              the predicted gain cannot appear. Tracked per draw;
  support     the correct answer must be reachable at all. If p* = 0, no
              search policy helps. Exposed as the oracle ceiling.

AND IT PREDICTS ITS OWN GAIN BEFORE IT RUNS
───────────────────────────────────────────
``predict_distinct_advantage`` computes, from a pilot sample alone, how many
DISTINCT candidates each policy will examine. That needs no knowledge of p*
— only the empirical mass profile — so it is checkable in the same run that
produces the outcome. A mechanism whose measured effect matches its
predicted effect is not just working; it is understood. A mechanism whose
measured effect misses its prediction tells you which premise broke.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.brain.llm.latent_cortex.commitment_ratchet import (
    CommitmentRatchet,
    Constraint,
    ConstraintKind,
    _normalize,
)

EXCLUSION_SCHEMA = "aura.rlc.sequential_exclusion.v2"
MAX_GENERATIONS_PER_VERIFIER_CALL = 4

#: Below this, the distribution is flat enough that i.i.d. sampling already
#: covers the space and exclusion has little to remove. Reported, not
#: enforced: a policy that quietly declines to run is a policy that cannot
#: be measured.
PEAKEDNESS_FLOOR = 0.15


class DrawOutcome(StrEnum):
    """What a verifier said about one drawn candidate."""

    #: Verified correct. The search is over.
    ACCEPTED = "accepted"
    #: Deterministically refuted. This is what earns an exclusion.
    REFUTED = "refuted"
    #: The verifier could not decide. NOT grounds for exclusion — removing
    #: what we could not check is how a search becomes a random walk.
    UNDECIDED = "undecided"
    #: The model redrew something already excluded. Not evidence about the
    #: answer; evidence about compliance.
    NONCOMPLIANT = "noncompliant"


# ────────────────────────────────────────────────────────── the arithmetic


def iid_success_probability(p_star: float, draws: int) -> float:
    """1 − (1 − p*)^N. The floor every other policy must beat."""
    p = min(1.0, max(0.0, float(p_star)))
    n = max(0, int(draws))
    return 1.0 - (1.0 - p) ** n


def exclusion_success_probability(
    p_star: float, excluded_masses: Sequence[float], draws: int
) -> float:
    """1 − Π (1 − p*/(1 − m_k)) over the accumulating excluded mass.

    ``excluded_masses`` is the mass each successive refutation removes, in
    the order it is removed. Once the remaining mass would be exhausted the
    probability saturates at 1.0 — at that point the correct answer is all
    that is left to draw.
    """
    p = min(1.0, max(0.0, float(p_star)))
    if p <= 0.0:
        return 0.0
    survival = 1.0
    removed = 0.0
    for index in range(max(0, int(draws))):
        remaining = 1.0 - removed
        if remaining <= p:
            return 1.0
        survival *= 1.0 - (p / remaining)
        if index < len(excluded_masses):
            removed = min(removed + max(0.0, float(excluded_masses[index])), 1.0 - p)
    return 1.0 - survival


def expected_distinct_iid(masses: Sequence[float], draws: int) -> float:
    """E[#distinct] = Σ_i (1 − (1 − p_i)^N).

    The quantity the whole argument turns on, and it needs no knowledge of
    which answer is correct — only the shape of the distribution. That is
    what makes the prediction checkable in the same run as the outcome.
    """
    n = max(0, int(draws))
    return sum(1.0 - (1.0 - min(1.0, max(0.0, float(m)))) ** n for m in masses)


def peakedness(masses: Sequence[float]) -> float:
    """Herfindahl concentration Σ p_i², in (0, 1].

    1.0 is a point mass (every draw identical — the 0.9994 case); near 0 is
    uniform. This is the single number that says how much exclusion has to
    gain, and it is measurable from a pilot sample before committing budget.
    """
    values = [min(1.0, max(0.0, float(m))) for m in masses]
    return sum(value * value for value in values)


def estimate_mass_profile(samples: Sequence[str]) -> list[float]:
    """Empirical mass of each distinct answer, largest first.

    Plain relative frequency. Deliberately NOT smoothed: a Good-Turing style
    correction would move mass onto unseen answers and make the predicted
    i.i.d. coverage look better than the sample supports, which flatters the
    baseline rather than the treatment — but it is still an invented number,
    and this module's whole claim rests on the prediction being honest.
    """
    normalised = [_normalize(sample) for sample in samples if str(sample).strip()]
    if not normalised:
        return []
    total = float(len(normalised))
    counts = Counter(normalised)
    return sorted((count / total for count in counts.values()), reverse=True)


@dataclass(frozen=True)
class DistinctAdvantage:
    """The prediction, made before the outcome is known."""

    draws: int
    distinct_iid: float
    distinct_exclusion: float
    peakedness: float
    pilot_samples: int
    pilot_distinct: int

    @property
    def advantage(self) -> float:
        """Extra distinct candidates exclusion is predicted to examine."""
        return self.distinct_exclusion - self.distinct_iid

    @property
    def advantage_ratio(self) -> float:
        if self.distinct_iid <= 0:
            return 0.0
        return self.distinct_exclusion / self.distinct_iid

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EXCLUSION_SCHEMA,
            "draws": self.draws,
            "expected_distinct_iid": round(self.distinct_iid, 4),
            "expected_distinct_exclusion": round(self.distinct_exclusion, 4),
            "advantage": round(self.advantage, 4),
            "advantage_ratio": round(self.advantage_ratio, 4),
            "peakedness": round(self.peakedness, 4),
            "pilot_samples": self.pilot_samples,
            "pilot_distinct": self.pilot_distinct,
            "worth_running": self.peakedness >= PEAKEDNESS_FLOOR,
            # Said plainly, because this is the number the campaign lives or
            # dies on and it must not be reconstructed after the fact.
            "prediction": (
                f"exclusion examines ~{self.distinct_exclusion:.1f} distinct "
                f"candidates in {self.draws} draws where i.i.d. examines "
                f"~{self.distinct_iid:.1f}"
            ),
        }


def predict_distinct_advantage(
    pilot_samples: Sequence[str], *, draws: int, compliance: float = 1.0
) -> DistinctAdvantage:
    """Predict the coverage gain from a pilot sample, before spending budget.

    ``compliance`` discounts the exclusion arm by the rate at which the model
    is expected to honour an exclusion. At compliance 0 the two policies
    coincide, which is the correct prediction: an exclusion the model ignores
    is not an exclusion.
    """
    masses = estimate_mass_profile(pilot_samples)
    n = max(0, int(draws))
    iid = expected_distinct_iid(masses, n)
    honoured = min(1.0, max(0.0, float(compliance)))
    # Exclusion examines N distinct candidates when honoured, and falls back
    # to the i.i.d. count when it is not.
    exclusion = honoured * float(n) + (1.0 - honoured) * iid
    return DistinctAdvantage(
        draws=n,
        distinct_iid=iid,
        distinct_exclusion=max(iid, exclusion),
        peakedness=peakedness(masses),
        pilot_samples=len([s for s in pilot_samples if str(s).strip()]),
        pilot_distinct=len(masses),
    )


# ──────────────────────────────────────────────────────────── the policy


@dataclass
class DrawRecord:
    """One draw, its verdict, and what it cost or bought."""

    index: int
    candidate: str
    outcome: DrawOutcome
    excluded_mass_estimate: float = 0.0
    was_duplicate_of_excluded: bool = False
    exclusions_active: bool = False
    verifier_called: bool = False
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "candidate": self.candidate[:160],
            "outcome": self.outcome.value,
            "excluded_mass_estimate": round(self.excluded_mass_estimate, 4),
            "noncompliant": self.was_duplicate_of_excluded,
            "exclusions_active": self.exclusions_active,
            "verifier_called": self.verifier_called,
            "detail": self.detail[:200],
        }


@dataclass
class ExclusionResult:
    """The search's outcome and the evidence for every premise."""

    answer: str | None
    draws: list[DrawRecord] = field(default_factory=list)
    prediction: DistinctAdvantage | None = None
    gold_exclusions: int = 0
    ratchet_receipt: dict[str, Any] = field(default_factory=dict)

    @property
    def distinct_examined(self) -> int:
        return len(
            {
                _normalize(row.candidate)
                for row in self.draws
                if row.candidate and row.verifier_called
            }
        )

    @property
    def verifier_calls(self) -> int:
        return sum(row.verifier_called for row in self.draws)

    @property
    def rejected_redraws(self) -> int:
        return sum(row.was_duplicate_of_excluded for row in self.draws)

    @property
    def compliance(self) -> float:
        """Fraction of post-exclusion draws that honoured the exclusions.

        Draw 0 has nothing to comply with and is excluded from the
        denominator; counting it would inflate compliance toward 1 for free.
        """
        eligible = [row for row in self.draws if row.exclusions_active]
        if not eligible:
            return 1.0
        honoured = sum(1 for row in eligible if not row.was_duplicate_of_excluded)
        return honoured / len(eligible)

    def to_dict(self) -> dict[str, Any]:
        predicted = self.prediction.distinct_exclusion if self.prediction else None
        return {
            "schema": EXCLUSION_SCHEMA,
            "answer": self.answer,
            "solved": self.answer is not None,
            "draws": [row.to_dict() for row in self.draws],
            "distinct_examined": self.distinct_examined,
            "generations": len(self.draws),
            "verifier_calls": self.verifier_calls,
            "rejected_redraws": self.rejected_redraws,
            "compliance": round(self.compliance, 4),
            # The premise that fails silently. One gold exclusion means the
            # verifier removed the truth, and no amount of search recovers
            # from that — it is a defect report, not a metric.
            "gold_exclusions": self.gold_exclusions,
            "verifier_sound": self.gold_exclusions == 0,
            "prediction": self.prediction.to_dict() if self.prediction else None,
            "predicted_distinct": predicted,
            "prediction_error": (
                None if predicted is None else round(self.distinct_examined - predicted, 4)
            ),
            "ratchet": self.ratchet_receipt,
        }


def run_sequential_exclusion(
    objective: str,
    *,
    draw: Callable[[str, str], str],
    verify: Callable[[str, str], tuple[DrawOutcome, str]],
    max_draws: int = 8,
    pilot_samples: Sequence[str] = (),
    gold_answer: str | None = None,
    ratchet: CommitmentRatchet | None = None,
) -> ExclusionResult:
    """Draw, verify, exclude, redraw — until verified or out of budget.

    ``draw(objective, requirement_block) -> candidate`` is the caller's model
    call. Refuted answers are never named in that block: the measured winning
    policy draws blindly, rejects a duplicate locally, and redraws without a
    verifier call. ``max_draws`` therefore budgets verifier calls; generation
    gets bounded headroom so duplicates do not consume the resource the A/B
    result held equal.

    ``gold_answer`` is optional and used ONLY to detect the verifier
    refuting a correct answer. It never influences a draw or a verdict; it
    is instrumentation for the premise, and a search that consulted it to
    decide would be measuring itself.
    """
    ratchet = ratchet if ratchet is not None else CommitmentRatchet()
    result = ExclusionResult(answer=None)
    if pilot_samples:
        result.prediction = predict_distinct_advantage(pilot_samples, draws=max_draws)

    excluded: set[str] = set()
    gold = _normalize(gold_answer) if gold_answer else ""

    verifier_budget = max(1, int(max_draws))
    max_generations = verifier_budget * MAX_GENERATIONS_PER_VERIFIER_CALL
    generations = 0
    while result.verifier_calls < verifier_budget and generations < max_generations:
        index = generations
        generations += 1
        exclusions_active = bool(excluded)
        try:
            candidate = str(draw(objective, ratchet.conditioning_block()) or "")
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            result.draws.append(
                DrawRecord(
                    index=index,
                    candidate="",
                    outcome=DrawOutcome.UNDECIDED,
                    exclusions_active=exclusions_active,
                    detail=f"draw failed: {type(exc).__name__}: {exc}",
                )
            )
            continue

        normalised = _normalize(candidate)
        if not normalised:
            result.draws.append(
                DrawRecord(
                    index=index,
                    candidate="",
                    outcome=DrawOutcome.UNDECIDED,
                    exclusions_active=exclusions_active,
                    detail="empty draw",
                )
            )
            continue

        if normalised in excluded:
            # The model ignored an exclusion. Recorded and NOT re-verified:
            # spending a verifier call on an answer already refuted is the
            # duplicate work exclusion exists to stop.
            result.draws.append(
                DrawRecord(
                    index=index,
                    candidate=candidate,
                    outcome=DrawOutcome.NONCOMPLIANT,
                    was_duplicate_of_excluded=True,
                    exclusions_active=True,
                    detail="redrew an excluded answer",
                )
            )
            continue

        try:
            outcome, detail = verify(objective, candidate)
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            outcome, detail = DrawOutcome.UNDECIDED, f"{type(exc).__name__}: {exc}"

        record = DrawRecord(
            index=index,
            candidate=candidate,
            outcome=outcome,
            exclusions_active=exclusions_active,
            verifier_called=True,
            detail=str(detail),
        )

        if outcome is DrawOutcome.ACCEPTED:
            result.draws.append(record)
            result.answer = candidate
            break

        if outcome is DrawOutcome.REFUTED:
            if gold and normalised == gold:
                # The premise just broke. Surfaced loudly: every later draw
                # in this episode is searching a space the truth is no
                # longer in.
                result.gold_exclusions += 1
                record.detail = f"GOLD EXCLUDED — verifier unsound: {record.detail}"
            excluded.add(normalised)
            receipt = ratchet.commit(
                Constraint(
                    kind=ConstraintKind.EXCLUDES,
                    subject=candidate.strip()[:120],
                    source="sequential_exclusion",
                    step=index,
                )
            )
            record.excluded_mass_estimate = float(receipt.narrowing or 0.0)

        # UNDECIDED excludes nothing. An answer the verifier could not
        # decide has not been shown wrong, and removing it would be removing
        # what we failed to check.
        result.draws.append(record)

    result.ratchet_receipt = ratchet.receipt()
    return result


def compare_to_iid(result: ExclusionResult) -> dict[str, Any]:
    """Did the measured coverage match the predicted coverage?

    This is the strongest evidence the mechanism can produce short of an
    end-to-end win: a quantity predicted from the pilot alone, then measured.
    Agreement means the model of WHY is right, so the effect is expected to
    transfer. Disagreement names which premise broke — compliance is on the
    same receipt.
    """
    if result.prediction is None:
        return {
            "schema": EXCLUSION_SCHEMA,
            "comparable": False,
            "reason": "no pilot sample, so no prediction was made",
        }
    predicted = result.prediction.distinct_exclusion
    measured = float(result.distinct_examined)
    baseline = result.prediction.distinct_iid
    return {
        "schema": EXCLUSION_SCHEMA,
        "comparable": True,
        "predicted_distinct": round(predicted, 4),
        "measured_distinct": measured,
        "iid_baseline_distinct": round(baseline, 4),
        "beat_iid_baseline": measured > baseline,
        "prediction_error": round(measured - predicted, 4),
        "relative_error": (round(abs(measured - predicted) / predicted, 4) if predicted else None),
        "compliance": round(result.compliance, 4),
        "verifier_sound": result.gold_exclusions == 0,
        "diagnosis": _diagnose(result, predicted, measured, baseline),
    }


def _diagnose(result: ExclusionResult, predicted: float, measured: float, baseline: float) -> str:
    if result.gold_exclusions:
        return (
            "verifier refuted a correct answer; the truth was excluded and no "
            "search policy recovers from that. Fix soundness before reading "
            "any other number here."
        )
    if result.compliance < 0.7:
        return (
            f"compliance {result.compliance:.0%}: the model kept redrawing "
            "excluded answers, so exclusion was nominal. The coverage gain "
            "cannot appear until the conditioning actually binds."
        )
    if measured <= baseline:
        return (
            "coverage did not beat the i.i.d. baseline despite compliance — "
            "the pilot mass profile did not describe the sampling behaviour "
            "under conditioning. Re-estimate the profile WITH the "
            "conditioning block present."
        )
    if abs(measured - predicted) <= max(1.0, 0.2 * predicted):
        return (
            "measured coverage matches prediction: the mechanism is behaving "
            "as modelled, and the gain has an explanation rather than a "
            "correlation."
        )
    return (
        "coverage beat the baseline but missed the prediction; the mass "
        "profile is mis-estimated even though the mechanism binds."
    )


__all__ = [
    "EXCLUSION_SCHEMA",
    "PEAKEDNESS_FLOOR",
    "DistinctAdvantage",
    "DrawOutcome",
    "DrawRecord",
    "ExclusionResult",
    "compare_to_iid",
    "estimate_mass_profile",
    "exclusion_success_probability",
    "expected_distinct_iid",
    "iid_success_probability",
    "peakedness",
    "predict_distinct_advantage",
    "run_sequential_exclusion",
]
