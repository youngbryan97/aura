"""Monotonic, verifier-grounded acceptance rule for a second (Nth) pass.

The measured failure mode of naive self-correction is not that models cannot
correct — it is that, without an oracle, a "second pass" flips correct
answers to incorrect MORE often than the reverse, and most of the time just
restates the first answer. Extra compute becomes longer hallucination rather
than more reasoning.

The fix is not "try harder." It is a decision rule: a revised answer replaces
the current one ONLY when the evidence for the challenger clearly exceeds the
evidence for the incumbent. This module is that rule, and nothing else.

    accept challenger  ⇔  LCB(Q(challenger)) > UCB(Q(incumbent)) + margin

``Q`` is a quality estimate grounded in Aura's real truth engines
(:class:`VerificationResult`), and the confidence half-width around it is
governed by the verifier's MEASURED reliability from the Verifier Foundry —
a proven verifier yields a tight interval that can decisively displace a
weaker answer, an unmeasured or leaky one yields a wide interval that cannot.
Uncertainty, ties, and unchecked evidence all KEEP the incumbent. That
default is the whole point: it makes additional passes safe, so they can be
run freely to search for a genuinely better answer without risking the good
one already in hand.

This is deliberately a pure function over verdicts — no model, no I/O, no
generation. It is the gate the reasoning amplifier applies when an incumbent
answer exists, and the gate the curriculum/deliberation loops apply between
successive attempts at the same problem.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# Tunables, chosen to be conservative (favor the incumbent under doubt).
# The interval half-width for a check we trust completely; scaled UP by a
# verifier's unreliability.
_BASE_HALF_WIDTH = 0.12
# Extra half-width charged to an UNCHECKED verdict: a check that did not run
# is not weak evidence, it is the absence of evidence.
_UNCHECKED_PENALTY = 0.5
# The challenger's lower bound must beat the incumbent's upper bound by at
# least this much — a decision margin that stops interval-boundary jitter
# from churning the answer.
_DEFAULT_MARGIN = 0.05
# Below this measured reliability a verifier cannot, on its own, DECISIVELY
# displace a clean incumbent no matter how high its raw score.
_MIN_RELIABILITY_TO_DISPLACE = 0.25


class RevisionVerdict(StrEnum):
    ACCEPT_CHALLENGER = "accept_challenger"
    KEEP_INCUMBENT = "keep_incumbent"
    ADOPT_FIRST = "adopt_first"  # no incumbent existed


@dataclass(frozen=True)
class QualityBound:
    """A verifier-grounded quality estimate with a confidence half-width."""

    point: float
    half_width: float
    checked: bool
    hard_ok: bool
    reliability: float

    @property
    def lower(self) -> float:
        return max(0.0, self.point - self.half_width)

    @property
    def upper(self) -> float:
        return min(1.0, self.point + self.half_width)

    def to_dict(self) -> dict[str, Any]:
        return {
            "point": round(self.point, 4),
            "lower": round(self.lower, 4),
            "upper": round(self.upper, 4),
            "checked": self.checked,
            "hard_ok": self.hard_ok,
            "reliability": round(self.reliability, 4),
        }


@dataclass(frozen=True)
class RevisionDecision:
    verdict: RevisionVerdict
    reason: str
    incumbent: QualityBound | None
    challenger: QualityBound
    margin: float
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def accept(self) -> bool:
        return self.verdict in (
            RevisionVerdict.ACCEPT_CHALLENGER,
            RevisionVerdict.ADOPT_FIRST,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "accept": self.accept,
            "reason": self.reason,
            "margin": round(self.margin, 4),
            "incumbent": self.incumbent.to_dict() if self.incumbent else None,
            "challenger": self.challenger.to_dict(),
            **({"detail": self.detail} if self.detail else {}),
        }


def _coerce01(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return max(0.0, min(1.0, result))


def quality_bound(
    verdict: Any,
    *,
    reliability: float = 0.5,
) -> QualityBound:
    """Translate a VerificationResult-shaped verdict into a bounded quality.

    ``verdict`` needs the duck-typed fields ``ok``, ``checked``, ``score``.
    ``reliability`` is the verifier's measured accuracy lower bound (from the
    Verifier Foundry's ``weight_for``); it tightens the interval as the
    verifier earns trust. A ``None`` verdict is treated as the weakest
    possible evidence (unchecked, not-ok, zero score).
    """
    reliability = _coerce01(reliability, 0.5)
    if verdict is None:
        return QualityBound(
            point=0.0,
            half_width=1.0,
            checked=False,
            hard_ok=False,
            reliability=reliability,
        )
    hard_ok = bool(getattr(verdict, "ok", False))
    checked = bool(getattr(verdict, "checked", False))
    score = _coerce01(getattr(verdict, "score", 0.5), 0.5)

    # A provable failure is a hard ceiling on quality regardless of the soft
    # score: an answer a real check REJECTED is not a good answer.
    if checked and not hard_ok:
        point = min(score, 0.15)
    elif not checked:
        # No check ran: the point estimate carries no positive evidence, so it
        # sits at the raw soft score but with a very wide interval.
        point = score
    else:
        point = score

    # Interval half-width: narrow when the verifier is proven and a check
    # actually ran; wide when unmeasured/leaky or unchecked.
    unreliability = 1.0 - reliability
    half = _BASE_HALF_WIDTH + (0.6 * unreliability)
    if not checked:
        half += _UNCHECKED_PENALTY
    return QualityBound(
        point=point,
        half_width=min(1.0, half),
        checked=checked,
        hard_ok=hard_ok,
        reliability=reliability,
    )


def decide_revision(
    incumbent_verdict: Any,
    challenger_verdict: Any,
    *,
    incumbent_reliability: float = 0.5,
    challenger_reliability: float = 0.5,
    margin: float = _DEFAULT_MARGIN,
    has_incumbent: bool = True,
) -> RevisionDecision:
    """Decide whether the challenger answer should REPLACE the incumbent.

    Returns a :class:`RevisionDecision`. The rule, in order:

    1. No incumbent → adopt the challenger (nothing to regress from).
    2. Challenger provably fails a check while the incumbent is clean → keep
       the incumbent. A regression can never be "accepted."
    3. Incumbent is clean-checked, challenger is unchecked → keep the
       incumbent; unchecked is not evidence of improvement.
    4. Challenger cleanly checked, incumbent unchecked/failing → accept
       (the challenger has real evidence the incumbent lacks).
    5. Both clean-checked → accept ONLY when the challenger's quality lower
       bound clears the incumbent's upper bound by ``margin`` AND the
       challenger's verifier is reliable enough to displace.
    Otherwise keep the incumbent.
    """
    try:
        margin = float(margin)
    except (TypeError, ValueError):
        margin = _DEFAULT_MARGIN
    if not math.isfinite(margin) or margin < 0.0:
        margin = _DEFAULT_MARGIN

    challenger = quality_bound(challenger_verdict, reliability=challenger_reliability)

    if not has_incumbent:
        return RevisionDecision(
            verdict=RevisionVerdict.ADOPT_FIRST,
            reason="no_incumbent",
            incumbent=None,
            challenger=challenger,
            margin=margin,
        )

    incumbent = quality_bound(incumbent_verdict, reliability=incumbent_reliability)

    # 2. A clean incumbent is never displaced by a provably-failing challenger.
    if incumbent.hard_ok and incumbent.checked and challenger.checked and not challenger.hard_ok:
        return RevisionDecision(
            RevisionVerdict.KEEP_INCUMBENT,
            "challenger_failed_verification",
            incumbent,
            challenger,
            margin,
        )

    # 3. A clean-checked incumbent is not displaced by an UNCHECKED challenger.
    if incumbent.hard_ok and incumbent.checked and not challenger.checked:
        return RevisionDecision(
            RevisionVerdict.KEEP_INCUMBENT,
            "challenger_unchecked_incumbent_verified",
            incumbent,
            challenger,
            margin,
        )

    # 4. Challenger has a clean real check the incumbent lacks.
    incumbent_clean = incumbent.hard_ok and incumbent.checked
    challenger_clean = challenger.hard_ok and challenger.checked
    if challenger_clean and not incumbent_clean:
        return RevisionDecision(
            RevisionVerdict.ACCEPT_CHALLENGER,
            "challenger_verified_incumbent_not",
            incumbent,
            challenger,
            margin,
        )

    # 5. Both clean (or both unclean): the confidence-bound comparison.
    if challenger_clean and incumbent_clean:
        if challenger.reliability < _MIN_RELIABILITY_TO_DISPLACE:
            return RevisionDecision(
                RevisionVerdict.KEEP_INCUMBENT,
                "challenger_verifier_too_unreliable_to_displace",
                incumbent,
                challenger,
                margin,
                detail={"min_reliability": _MIN_RELIABILITY_TO_DISPLACE},
            )
        if challenger.lower > incumbent.upper + margin:
            return RevisionDecision(
                RevisionVerdict.ACCEPT_CHALLENGER,
                "challenger_lower_bound_clears_incumbent",
                incumbent,
                challenger,
                margin,
                detail={
                    "challenger_lower": round(challenger.lower, 4),
                    "incumbent_upper": round(incumbent.upper, 4),
                },
            )
        return RevisionDecision(
            RevisionVerdict.KEEP_INCUMBENT,
            "insufficient_evidence_to_revise",
            incumbent,
            challenger,
            margin,
            detail={
                "challenger_lower": round(challenger.lower, 4),
                "incumbent_upper": round(incumbent.upper, 4),
            },
        )

    # Neither is cleanly verified: this is genuinely uncertain ground. Only
    # move off the incumbent for a clearly-larger point estimate past the
    # margin — but a first answer is worth as much as a second guess here, so
    # the incumbent still wins ties.
    if challenger.point > incumbent.point + max(margin, 0.15):
        return RevisionDecision(
            RevisionVerdict.ACCEPT_CHALLENGER,
            "both_unverified_challenger_materially_higher",
            incumbent,
            challenger,
            margin,
        )
    return RevisionDecision(
        RevisionVerdict.KEEP_INCUMBENT,
        "both_unverified_no_clear_gain",
        incumbent,
        challenger,
        margin,
    )


@dataclass
class DeliberationResult:
    """The outcome of a bounded, revision-gated multi-pass over one problem."""

    answer: str
    verdict: Any
    passes: int
    accepted_revisions: int
    rejected_revisions: int
    verified: bool
    trail: list[dict[str, Any]] = field(default_factory=list)
    #: Distinct answers actually examined. Under i.i.d. sampling from a
    #: peaked model this is far below ``passes`` — best-of-8 is best-of-2 —
    #: and it is the quantity a search improvement has to move.
    distinct_answers: int = 0
    #: Candidates excluded after being REFUTED with a checked verdict.
    exclusions: int = 0
    #: Draws discarded for repeating an already-refuted answer. These cost a
    #: generation and SAVE a verifier call, which is the trade the measured
    #: gain comes from — verifier calls fell 3.97 -> 3.19 while the solve
    #: rate rose.
    rejected_redraws: int = 0
    #: Total model generations, including duplicates rejected before a
    #: verifier call. This keeps the compute trade visible without pretending
    #: a discarded redraw was a revision decision.
    generations: int = 0
    #: Empty generations rejected before verification.
    generation_failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "passes": self.passes,
            "accepted_revisions": self.accepted_revisions,
            "rejected_revisions": self.rejected_revisions,
            "verified": self.verified,
            "distinct_answers": self.distinct_answers,
            "exclusions": self.exclusions,
            "rejected_redraws": self.rejected_redraws,
            "generations": self.generations,
            "generation_failures": self.generation_failures,
            # passes is what was SPENT; distinct is what was examined. A gap
            # between them is duplicate work, and it is the whole reason the
            # exclusion path exists.
            "duplicate_passes": max(0, self.passes - self.distinct_answers),
            "trail": self.trail,
        }


def _accepts_conditioning(solve: Any) -> bool:
    """Does this solver accept the exclusion block as a second argument?

    Detected once, from the signature, rather than by calling and catching
    TypeError — a TypeError raised INSIDE a two-argument solver would then
    be misread as "it only takes one", and every later pass would silently
    drop the exclusions while the receipt still claimed they were applied.
    """
    import inspect

    try:
        signature = inspect.signature(solve)
    # not a failure: a solver whose signature cannot be read is not one this can
    # prove passes its exclusions, and False refuses.
    except (TypeError, ValueError):
        return False
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    if any(
        parameter.kind is parameter.VAR_POSITIONAL for parameter in signature.parameters.values()
    ):
        return True
    return len(positional) >= 2


async def deliberate_best_of(
    solve: Any,
    verify: Any,
    *,
    max_passes: int = 3,
    reliability_of: Any = None,
    margin: float = _DEFAULT_MARGIN,
    stop_when_verified: bool = False,
    exclude_refuted: bool = True,
) -> DeliberationResult:
    """Run up to ``max_passes`` INDEPENDENT solves, keeping an answer only when
    the revision gate says its evidence clearly beats the one in hand.

    This is the concrete "make a second pass safe" loop: each pass is a fresh
    ``await solve(pass_index)`` (blind — it does not see prior candidates, so
    there is no chain to rationalize toward), scored by ``await verify(answer)``
    (grounded in real truth engines), and adopted only through
    :func:`decide_revision`. The invariant is monotonic: the returned answer
    is never worse-evidenced than the best seen, so spending more passes can
    only help.

    ``reliability_of`` is an optional ``(verdict) -> float`` supplying the
    verifier's measured reliability (e.g. from the Verifier Foundry) so the
    confidence bounds are grounded in track record; it defaults to a neutral
    0.5. ``solve``/``verify`` may be sync or async.

    ``exclude_refuted`` turns independent sampling into sampling WITHOUT
    REPLACEMENT, by REJECTION: a pass that re-produces an answer already
    refuted with a checked verdict is discarded and redrawn, without
    spending a verifier call on a question already answered.

    It does NOT describe the excluded answers in the prompt. That was tried
    and measured, and it lost. Telling the model "not 483060, not 483000"
    perturbs the distribution it is supposed to be restricting and anchors
    on the very values being excluded: 46.9% vs 48.1% for plain i.i.d.,
    with distinct coverage FALLING. The theorem says to draw from p
    restricted to A\\R — rejection sampling is that; prompting is a proxy
    for it, and a bad one.

    This does not reintroduce the chain the blindness exists to prevent. The
    hazard of showing a prior candidate is rationalisation — the model sees
    a plausible answer and agrees with it. A REFUTED candidate is the
    opposite: it is removed, not offered, and only ever on a checked
    refutation. Unrefuted and undecided candidates are never shown, so the
    passes stay blind exactly where blindness was buying something.

    What it buys is coverage. Under i.i.d. sampling from a peaked
    distribution most passes re-derive the same answer, so best-of-8 is
    best-of-2 in anything that matters — measured at 2.58 distinct answers
    per 8 draws on a real model. Each draw after an exclusion has success
    probability p*/(1 − m) instead of p*, which dominates for every N and
    every distribution.

    Measured end to end 2026-08-09, Qwen2.5-1.5B, 160 paired tasks:
    48.1% -> 59.4% solved (z=2.02, p=0.044) on FEWER verifier calls
    (3.97 -> 3.19). See ``core.brain.llm.latent_cortex.sequential_exclusion``
    for the arithmetic and the three premises that can break it, and
    docs/RLC_COMMITMENT_SEARCH.md for what the measurement does not cover.
    """
    import inspect

    from core.brain.llm.latent_cortex.commitment_ratchet import (
        CommitmentRatchet,
        Constraint,
        ConstraintKind,
        _normalize,
    )

    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _reliability(verdict: Any) -> float:
        if reliability_of is None:
            return 0.5
        try:
            return _coerce01(reliability_of(verdict), 0.5)
        except (TypeError, ValueError, AttributeError):
            return 0.5

    try:
        passes = max(1, int(max_passes))
    except (TypeError, ValueError):
        passes = 3

    best_answer = ""
    best_verdict: Any = None
    best_reliability = 0.5
    have_best = False
    accepted = 0
    rejected_revisions = 0
    trail: list[dict[str, Any]] = []
    ratchet = CommitmentRatchet()
    seen_answers: set[str] = set()
    exclusions = 0
    solve_takes_conditioning = _accepts_conditioning(solve)

    refuted_answers: set[str] = set()
    rejected_redraws = 0
    generation_failures = 0
    index = -1
    generations = 0
    # Rejection sampling needs headroom to redraw. Bounded, so a model that
    # refuses to move cannot spin: it falls back to spending the budget.
    max_generations = passes * 3
    while len(trail) < passes and generations < max_generations:
        generations += 1
        index += 1
        if solve_takes_conditioning:
            # Rejection sampling must not name the excluded answers. The
            # measured prompt-conditioned arm lost to ordinary i.i.d.; the
            # winning arm draws blindly and rejects duplicates locally.
            raw = solve(index, "")
        else:
            raw = solve(index)
        answer = str(await _maybe_await(raw) or "").strip()
        if not answer:
            generation_failures += 1
            continue
        normalized_answer = _normalize(answer)
        if exclude_refuted and normalized_answer in refuted_answers:
            # Already checked and refuted. Redraw rather than paying a
            # verifier call to reach the same verdict twice.
            rejected_redraws += 1
            continue
        seen_answers.add(normalized_answer)
        verdict = await _maybe_await(verify(answer))
        reliability = _reliability(verdict)
        decision = decide_revision(
            best_verdict,
            verdict,
            incumbent_reliability=best_reliability,
            challenger_reliability=reliability,
            margin=margin,
            has_incumbent=have_best,
        )
        trail.append({"pass": index, **decision.to_dict()})
        if decision.accept:
            best_answer = answer
            best_verdict = verdict
            best_reliability = reliability
            have_best = True
            if decision.verdict is RevisionVerdict.ACCEPT_CHALLENGER:
                accepted += 1
        else:
            rejected_revisions += 1

        # Sampling without replacement, and ONLY on a checked refutation.
        # An unchecked failure is a verifier that could not decide, and
        # excluding what was never checked would remove the answer on the
        # strength of not having looked at it.
        if (
            exclude_refuted
            and getattr(verdict, "ok", None) is False
            and getattr(verdict, "checked", False) is True
        ):
            refuted_answers.add(normalized_answer)
            if ratchet.commit(
                Constraint(
                    kind=ConstraintKind.EXCLUDES,
                    subject=answer[:120],
                    source="deliberate_best_of",
                    step=index,
                )
            ).committed:
                exclusions += 1
                trail[-1]["excluded"] = True

        verified = bool(
            have_best
            and getattr(best_verdict, "ok", False)
            and getattr(best_verdict, "checked", False)
        )
        if stop_when_verified and verified:
            break

    verified = bool(
        have_best and getattr(best_verdict, "ok", False) and getattr(best_verdict, "checked", False)
    )
    return DeliberationResult(
        answer=best_answer,
        verdict=best_verdict,
        passes=len(trail),
        accepted_revisions=accepted,
        rejected_revisions=rejected_revisions,
        verified=verified,
        trail=trail,
        distinct_answers=len(seen_answers),
        exclusions=exclusions,
        rejected_redraws=rejected_redraws,
        generations=generations,
        generation_failures=generation_failures,
    )


__all__ = [
    "DeliberationResult",
    "QualityBound",
    "RevisionDecision",
    "RevisionVerdict",
    "decide_revision",
    "deliberate_best_of",
    "quality_bound",
]
