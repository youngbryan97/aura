"""Every bipartition of the core, scored on what the cut costs its future.

Five hundred and eleven cuts, and the system is worth its weakest one. That is
the whole difficulty: a minimum over five hundred and eleven noisy estimates is
biased downward by the width of its own search, and paying for enough precision
on all of them at once is not affordable.

The answer is sequential allocation rather than a shorter list. Every cut opens
with the same small set of anchors. A cut whose lower bound already clears the
sham floor is decided and stops drawing samples; the budget that would have
gone to it goes to the cuts still compatible with zero. What ends the run is
either every cut being decided or one of them staying undecided at the ceiling,
and the second of those is a result rather than a failure — it says the
experiment lacked the power to separate that partition, which is a different
statement from "the system is reducible there".

Nothing here deletes a cut for being inconvenient. `EXCLUDED` exists only to
record cuts that cannot be run at all, and it is empty.

The cut itself is not a regression ablation. `core.subject.clamp.compose`
builds it from two complementary clamped runs off one snapshot: the left side
evolving with the right held still, the right evolving with the left held
still, each side's columns taken from the run where it was free. Both sides
keep their internal dynamics and the cross-partition channels are the only
thing gone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from core.subject.intrinsic_v25 import (
    IntrinsicRateEstimate,
    bootstrap_rate_difference,
    intrinsic_rate_from_samples,
    paired_permutation_pvalue,
)
from core.subject.v25_runtime import bipartitions, collect_partition_samples

logger = logging.getLogger(__name__)

__all__ = [
    "CutVerdict",
    "SweepReport",
    "OPENING_ANCHORS",
    "ANCHOR_STEP",
    "decide_cut",
    "sweep_cuts",
]

#: Anchors every cut opens with, before any of them has earned more.
OPENING_ANCHORS: int = 8

#: Anchors added to an undecided cut on each further round.
ANCHOR_STEP: int = 8

#: Cuts that cannot be run at all. A cut is never removed for scoring badly.
EXCLUDED: frozenset[tuple[tuple[str, ...], tuple[str, ...]]] = frozenset()

#: The fewest anchors a cut can be scored from. The estimator cross-fits over
#: five folds and needs at least one row per fold on each side, so four anchors
#: produce no score at all — the first quick run came back "0 of 12 decided"
#: with nothing saying why. Below this the sweep refuses rather than running and
#: reporting nothing.
MINIMUM_ANCHORS: int = 5


@dataclass
class CutVerdict:
    """One partition, and whether the experiment could decide it."""

    left: tuple[str, ...]
    right: tuple[str, ...]
    anchors_used: int
    estimate: IntrinsicRateEstimate | None = None
    excess: float = 0.0
    lower_bound: float = 0.0
    p_value: float = 1.0
    decided: bool = False
    note: str = ""

    @property
    def name(self) -> str:
        return f"{''.join(self.left)}|{''.join(self.right)}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "cut": self.name,
            "anchors": self.anchors_used,
            "excess_rate": round(self.excess, 6),
            "lower_bound": round(self.lower_bound, 6),
            "p_value": round(self.p_value, 6),
            "raw_rate": None if self.estimate is None else round(self.estimate.raw_rate, 6),
            "sham_rate": None if self.estimate is None else round(self.estimate.sham_rate, 6),
            "decided": self.decided,
            "note": self.note,
        }


@dataclass
class SweepReport:
    """Every cut that was tried, and the weakest one that was decided."""

    tau_seconds: float
    #: True when only a sample of the cuts was scored. A screened sweep is a
    #: look, not a result: the score is the weakest cut and a sample has not
    #: found it.
    screened: bool = False
    cuts_in_full: int = 0
    #: Cuts the estimator could not score at all. Not the same as a cut that
    #: scored badly, and a sweep where this equals the cut count measured
    #: nothing rather than measuring zero.
    unscorable: int = 0
    verdicts: list[CutVerdict] = field(default_factory=list)
    #: Cuts still compatible with zero when the budget ran out. While this is
    #: not empty the irreducibility claim is UNRESOLVED, not false.
    undecided: list[str] = field(default_factory=list)
    anchors_spent: int = 0

    @property
    def weakest(self) -> CutVerdict | None:
        scored = [v for v in self.verdicts if v.estimate is not None]
        if not scored:
            return None
        return min(scored, key=lambda v: v.lower_bound)

    @property
    def irreducible(self) -> bool:
        """Every cut changed the future law, and none was left undecided.

        An intersection-union test: the claim is a conjunction over cuts, so
        one cut compatible with zero refuses it and no amount of margin
        elsewhere buys it back.
        """
        if self.screened:
            return False
        return bool(self.verdicts) and not self.undecided and all(
            v.decided and v.lower_bound > 0.0 for v in self.verdicts
        )

    def as_dict(self) -> dict[str, Any]:
        weakest = self.weakest
        ranked = sorted(
            (v for v in self.verdicts if v.estimate is not None),
            key=lambda v: v.lower_bound,
        )
        return {
            "tau_seconds": round(self.tau_seconds, 6),
            "cuts_tested": len(self.verdicts),
            "cuts_in_full": self.cuts_in_full,
            "screened": self.screened,
            "cuts_decided": sum(1 for v in self.verdicts if v.decided),
            "cuts_unscorable": self.unscorable,
            "measured_nothing": bool(
                self.verdicts and not any(v.estimate is not None for v in self.verdicts)
            ),
            "undecided": list(self.undecided),
            "anchors_spent": self.anchors_spent,
            "irreducible": self.irreducible,
            "weakest_cut": None if weakest is None else weakest.name,
            "weakest_lower_bound": None if weakest is None else round(weakest.lower_bound, 6),
            "weakest_excess": None if weakest is None else round(weakest.excess, 6),
            "cheapest": [v.as_dict() for v in ranked[:8]],
            "dearest": [v.as_dict() for v in ranked[-4:]],
            # Every cut that cleared its floor, by name. The graph one level up
            # is built from these, and a summary of the eight cheapest would
            # have built it from a twelfth of the evidence.
            "decided_cuts": [v.name for v in self.verdicts if v.decided],
        }


def decide_cut(
    samples: dict[str, np.ndarray],
    *,
    tau_seconds: float,
    draws: int = 200,
    permutation_draws: int = 199,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[IntrinsicRateEstimate, float, float, float]:
    """Score one cut: the excess rate, its lower bound, and a paired p-value.

    The lower bound is the alpha quantile of a paired bootstrap over the
    matched contexts, which is what a sequential rule needs: a cut is decided
    when its whole interval sits above the floor, not when its point estimate
    does.
    """
    context = samples.get("context")
    estimate = intrinsic_rate_from_samples(
        samples["intact"],
        samples["cut"],
        samples["sham_a"],
        samples["sham_b"],
        tau_seconds=tau_seconds,
        context=context,
        seed=seed,
    )
    spread = bootstrap_rate_difference(
        samples["intact"],
        samples["cut"],
        samples["sham_a"],
        samples["sham_b"],
        tau_seconds=tau_seconds,
        context=context,
        draws=draws,
        seed=seed + 1,
    )
    lower = float(np.quantile(spread, alpha))
    try:
        _observed, p_value = paired_permutation_pvalue(
            samples["intact"],
            samples["cut"],
            samples["sham_a"],
            samples["sham_b"],
            tau_seconds=tau_seconds,
            context=context,
            draws=permutation_draws,
            seed=seed + 2,
        )
    except ValueError:
        # Too few matched contexts for a randomization test. The bootstrap
        # still says something; the p-value says nothing and must not be
        # reported as though it did.
        p_value = 1.0
    return estimate, float(estimate.raw_rate - estimate.sham_rate), lower, float(p_value)


async def sweep_cuts(
    runtime: Any,
    anchors: Sequence[Any],
    conditions: Sequence[Any],
    *,
    tau_frames: int,
    tau_seconds: float,
    turns: int = 1,
    rounds: int = 3,
    alpha: float = 0.05,
    seed: int = 0,
    domains: Sequence[str] | None = None,
    screen: int = 0,
    on_progress: Any = None,
) -> SweepReport:
    """Every bipartition, with precision spent where the answer is still open.

    Round one gives every cut `OPENING_ANCHORS`. Each later round adds
    `ANCHOR_STEP` anchors to the cuts that are still compatible with zero and
    leaves the decided ones alone.
    """
    if len(anchors) < MINIMUM_ANCHORS:
        raise ValueError(
            f"{len(anchors)} anchors is fewer than the {MINIMUM_ANCHORS} the "
            "cross-fitted estimator needs; a sweep from this many scores no cut "
            "at all and would report an empty result as a measurement"
        )
    cuts = list(bipartitions(tuple(domains))) if domains else list(bipartitions())
    every_cut = len(cuts)
    screened = False
    if screen and 0 < screen < len(cuts):
        # A screening pass, and it says so. An exhaustive sweep of 511 cuts at
        # four runs an anchor is a long run, and there is a real use for a fast
        # look at where the weak cuts are before paying for one. What there is
        # no use for is a fast look reported as a result: the score is the
        # weakest cut, and a sample of cuts has not found it. So a screened
        # sweep never reads as irreducible and the runner refuses to call it
        # authoritative.
        #
        # The sample is a deterministic stride rather than a draw, so the same
        # seed screens the same cuts, and it walks the sizes evenly rather than
        # taking the first N — the cheapest cuts in practice are the lopsided
        # ones, and the first N of an ordered enumeration are all one shape.
        stride = max(1, len(cuts) // screen)
        cuts = cuts[:: stride][:screen]
        screened = True
    report = SweepReport(tau_seconds=float(tau_seconds), screened=screened, cuts_in_full=every_cut)
    verdicts: dict[str, CutVerdict] = {}
    for left, right in cuts:
        verdicts[f"{''.join(left)}|{''.join(right)}"] = CutVerdict(left=left, right=right, anchors_used=0)

    budget = OPENING_ANCHORS
    unscorable = 0
    for round_index in range(max(1, rounds)):
        pending = [v for v in verdicts.values() if not v.decided]
        if not pending:
            break
        take = min(len(anchors), budget)
        if take < 2:
            break
        chosen = list(anchors[:take])
        for position, verdict in enumerate(pending):
            samples = await collect_partition_samples(
                runtime,
                chosen,
                conditions,
                left=verdict.left,
                right=verdict.right,
                turns=turns,
                lags=(tau_frames,),
            )
            slot = samples[int(tau_frames)]
            report.anchors_spent += take
            verdict.anchors_used = take
            try:
                estimate, excess, lower, p_value = decide_cut(
                    slot,
                    tau_seconds=tau_seconds,
                    seed=seed + position,
                    alpha=alpha,
                )
            except ValueError as exc:
                # Recorded on the verdict AND counted, so a sweep where every
                # cut failed for the same reason says so instead of coming back
                # with an empty table.
                verdict.note = f"not enough matched contexts: {exc}"
                unscorable += 1
                continue
            verdict.estimate = estimate
            verdict.excess = excess
            verdict.lower_bound = lower
            verdict.p_value = p_value
            verdict.decided = lower > 0.0
            if on_progress is not None:
                on_progress(round_index, verdict)
        budget += ANCHOR_STEP

    report.verdicts = list(verdicts.values())
    report.undecided = sorted(v.name for v in report.verdicts if not v.decided)
    report.unscorable = unscorable
    if unscorable and not any(v.estimate is not None for v in report.verdicts):
        logger.warning(
            "every one of %d cuts was unscorable; the sweep measured nothing",
            unscorable,
        )
    return report
