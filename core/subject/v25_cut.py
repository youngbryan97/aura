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
import math
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
    "anchors_exchangeable",
    "merge_shard_payloads",
    "merge_sweeps",
    "shard_of",
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


@dataclass(frozen=True)
class RecordedRate:
    """The two rates of an estimate made in another process, read back from its report."""

    raw_rate: float
    sham_rate: float


def shard_of(cuts: Sequence[Any], index: int, count: int) -> list[tuple[int, Any]]:
    """Every `count`-th cut starting at `index`, each with its place in the full list.

    The place travels with the cut so a cut's bootstrap seed is a property of
    the cut, and a sharded sweep draws exactly what one process would have.
    """
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"shard {index} of {count} does not exist")
    return [(place, cut) for place, cut in enumerate(cuts) if place % count == index]


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

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> CutVerdict:
        """A verdict read back from `as_dict`, so shards run elsewhere can be merged.

        The Fisher-Rao estimates behind the rates are not carried across, and
        nothing downstream of a sweep reads them: the weakest cut, the decided
        set and the report read the rates, the bound and the decision. The rates
        come back on a record with the two fields the report prints.
        """
        left, _, right = str(row["cut"]).partition("|")
        estimate = None
        if row.get("raw_rate") is not None:
            estimate = RecordedRate(raw_rate=float(row["raw_rate"]), sham_rate=float(row.get("sham_rate") or 0.0))
        return cls(
            left=tuple(left),
            right=tuple(right),
            anchors_used=int(row.get("anchors", 0)),
            estimate=estimate,
            excess=float(row.get("excess_rate", 0.0)),
            lower_bound=float(row.get("lower_bound", 0.0)),
            p_value=float(row.get("p_value", 1.0)),
            decided=bool(row.get("decided", False)),
            note=str(row.get("note", "")),
        )

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
    #: "index/count" when this sweep scored one shard of the cuts. A shard on its
    #: own has not scored every cut and claims nothing until merged.
    shard: str = ""

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
        if self.screened or self.shard:
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
            "shard": self.shard,
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
            "verdicts": [v.as_dict() for v in self.verdicts],
        }


def merge_sweeps(shards: Sequence[dict[str, Any]], *, cuts_in_full: int) -> SweepReport:
    """One horizon's sweep from its shards, refused unless they cover every cut exactly once.

    Each shard is a `SweepReport.as_dict` for the same horizon. A cut scored by
    two shards, or by none, means the shards were not a partition of the cuts,
    and a merged report would claim a sweep nobody ran.
    """
    if not shards:
        raise ValueError("nothing to merge")
    taus = {round(float(s["tau_seconds"]), 6) for s in shards}
    if len(taus) != 1:
        raise ValueError(f"shards disagree about the horizon: {sorted(taus)}")
    verdicts: list[CutVerdict] = []
    seen: set[str] = set()
    for shard in shards:
        if shard.get("screened"):
            raise ValueError("a screened shard is a look, and a merge of looks is not a sweep")
        for row in shard.get("verdicts", []):
            if row["cut"] in seen:
                raise ValueError(f"cut {row['cut']} was scored by more than one shard")
            seen.add(row["cut"])
            verdicts.append(CutVerdict.from_dict(row))
    if len(seen) != cuts_in_full:
        raise ValueError(f"shards scored {len(seen)} of {cuts_in_full} cuts")
    return SweepReport(
        tau_seconds=float(shards[0]["tau_seconds"]),
        cuts_in_full=cuts_in_full,
        unscorable=sum(int(s.get("cuts_unscorable", 0)) for s in shards),
        verdicts=verdicts,
        undecided=sorted(v.name for v in verdicts if not v.decided),
        anchors_spent=sum(int(s.get("anchors_spent", 0)) for s in shards),
    )


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


async def sweep_cuts_over_lags(
    runtime: Any,
    anchors: Sequence[Any],
    conditions: Sequence[Any],
    *,
    lags: Sequence[int],
    frame_seconds: float,
    turns: int = 1,
    rounds: int = 3,
    alpha: float = 0.05,
    seed: int = 0,
    domains: Sequence[str] | None = None,
    screen: int = 0,
    shard: tuple[int, int] | None = None,
) -> dict[int, SweepReport]:
    """Every bipartition at every horizon, from one set of rollouts per cut.

    `sweep_cuts` scores one horizon, and the spectrum called it once per
    horizon, so every cut's four clamped arms were run again for every lag on
    the ladder. One rollout already holds every lag: `lag_vector` reads a frame
    index out of the same trajectory. Seven horizons cost seven times what one
    did, which put an exhaustive run on this host at days rather than hours.

    Here each round collects once per pending cut, at every horizon together,
    and decides each horizon on its own slot with its own seed. A cut keeps
    drawing anchors while any horizon it reached is undecided.

    A horizon the rollouts did not reach is reported as unreached and left
    undecided. `lag_vector` clamps a lag past the last recorded frame to that
    last frame, so without this every lag longer than a rollout would be scored
    as the same frame under a different name.
    """
    if len(anchors) < MINIMUM_ANCHORS:
        raise ValueError(
            f"{len(anchors)} anchors is fewer than the {MINIMUM_ANCHORS} the "
            "cross-fitted estimator needs; a sweep from this many scores no cut "
            "at all and would report an empty result as a measurement"
        )
    ladder = sorted({int(lag) for lag in lags})
    if not ladder:
        raise ValueError("a sweep needs at least one horizon to score")
    cuts = list(bipartitions(tuple(domains))) if domains else list(bipartitions())
    every_cut = len(cuts)
    screened = False
    if screen and 0 < screen < len(cuts):
        # The same deterministic stride `sweep_cuts` screens with, so a
        # screened look at the ladder covers the same cuts as before.
        stride = max(1, len(cuts) // screen)
        cuts = cuts[::stride][:screen]
        screened = True

    placed = shard_of(cuts, *shard) if shard is not None else list(enumerate(cuts))
    reports = {
        lag: SweepReport(
            tau_seconds=float(lag) * float(frame_seconds),
            screened=screened,
            cuts_in_full=every_cut,
            shard="" if shard is None else f"{shard[0]}/{shard[1]}",
        )
        for lag in ladder
    }
    place_of = {f"{''.join(left)}|{''.join(right)}": place for place, (left, right) in placed}
    cuts = [cut for _, cut in placed]
    names = [f"{''.join(left)}|{''.join(right)}" for left, right in cuts]
    verdicts = {
        lag: {
            name: CutVerdict(left=left, right=right, anchors_used=0)
            for name, (left, right) in zip(names, cuts, strict=True)
        }
        for lag in ladder
    }
    unreached: dict[int, set[str]] = {lag: set() for lag in ladder}
    unscorable: dict[int, int] = {lag: 0 for lag in ladder}

    def open_at(name: str, lag: int) -> bool:
        return not verdicts[lag][name].decided and name not in unreached[lag]

    budget = OPENING_ANCHORS
    for _round in range(max(1, rounds)):
        pending = [name for name in names if any(open_at(name, lag) for lag in ladder)]
        if not pending:
            break
        take = min(len(anchors), budget)
        if take < 2:
            break
        chosen = list(anchors[:take])
        for name in pending:
            # Seeded by the cut's place in the full list rather than its place
            # in this round's queue, which shrinks as cuts are decided and is
            # different in every shard.
            position = place_of[name]
            cut = verdicts[ladder[0]][name]
            samples = await collect_partition_samples(
                runtime,
                chosen,
                conditions,
                left=cut.left,
                right=cut.right,
                turns=turns,
                lags=tuple(ladder),
            )
            for lag in ladder:
                if not open_at(name, lag):
                    continue
                verdict = verdicts[lag][name]
                slot = samples[lag]
                reached = slot.get("reached")
                if reached is not None and float(np.min(reached)) < 1.0:
                    unreached[lag].add(name)
                    verdict.note = (
                        f"the rollouts ended before frame {lag}, so this horizon was not reached"
                    )
                    continue
                reports[lag].anchors_spent += take
                verdict.anchors_used = take
                try:
                    estimate, excess, lower, p_value = decide_cut(
                        slot,
                        tau_seconds=reports[lag].tau_seconds,
                        seed=seed + lag + position,
                        alpha=alpha,
                    )
                except ValueError as exc:
                    verdict.note = f"not enough matched contexts: {exc}"
                    unscorable[lag] += 1
                    continue
                verdict.estimate = estimate
                verdict.excess = excess
                verdict.lower_bound = lower
                verdict.p_value = p_value
                verdict.decided = lower > 0.0
        budget += ANCHOR_STEP

    for lag in ladder:
        report = reports[lag]
        report.verdicts = list(verdicts[lag].values())
        report.undecided = sorted(v.name for v in report.verdicts if not v.decided)
        report.unscorable = unscorable[lag]
        if unscorable[lag] and not any(v.estimate is not None for v in report.verdicts):
            logger.warning(
                "every one of %d cuts was unscorable at lag %d; the sweep measured nothing there",
                unscorable[lag],
                lag,
            )
    return reports


def anchors_exchangeable(
    reference: np.ndarray,
    other: np.ndarray,
    *,
    comparisons: int = 1,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Whether two anchor banks look drawn from one ordinary life, by a permutation test.

    A shard worker is its own process with its own organism, and two processes
    do not reach bit-identical states even from one seed. So a shard scores its
    cuts from its own anchors, and the merge is only honest if those anchors
    are the same kind of state the coordinator forked from. The statistic is
    the squared distance between the two banks' mean states, each column scaled
    by the pooled spread; the null permutes which bank each anchor came from.

    One test runs per shard, so the level is alpha divided by the number of
    comparisons. The draws are the fewest at which the smallest attainable
    p-value is a twentieth of that level, so a p-value near the level is
    resolved rather than rounded.
    """
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(other, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or len(a) < 2 or len(b) < 2:
        return {
            "measured": False,
            "exchangeable": False,
            "why": "the two banks are not both at least two anchors of the same width",
        }
    pooled = np.vstack([a, b])
    spread = pooled.std(axis=0)
    live = spread > 1e-12
    level = float(alpha) / max(1, int(comparisons))
    if not live.any():
        return {
            "measured": True,
            "exchangeable": True,
            "p_value": 1.0,
            "level": level,
            "why": "no column moves in either bank",
        }
    scaled = pooled[:, live] / spread[live]
    split = len(a)

    def statistic(rows: np.ndarray) -> float:
        return float(np.sum((rows[:split].mean(axis=0) - rows[split:].mean(axis=0)) ** 2))

    observed = statistic(scaled)
    draws = math.ceil(20.0 / level)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(draws):
        if statistic(scaled[rng.permutation(len(scaled))]) >= observed - 1e-15:
            exceed += 1
    p_value = (exceed + 1.0) / (draws + 1.0)
    exchangeable = p_value >= level
    return {
        "measured": True,
        "exchangeable": exchangeable,
        "p_value": round(p_value, 6),
        "level": round(level, 6),
        "draws": draws,
        "statistic": round(observed, 6),
        "reference_anchors": int(len(a)),
        "shard_anchors": int(len(b)),
        "why": (
            "the shard's anchors are the kind of state the coordinator forked from"
            if exchangeable
            else f"the shard's mean anchor state differs from the coordinator's at p = {p_value:.4g}"
        ),
    }


def merge_shard_payloads(
    payloads: Sequence[dict[str, Any]],
    *,
    reference_anchors: np.ndarray,
    cuts_in_full: int,
    lags: Sequence[int],
    frame_seconds: float,
    support: Sequence[str],
    conditions: Sequence[str],
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[dict[int, SweepReport], dict[str, Any]]:
    """The whole sweep from its shard files, and the check that let them be merged.

    Refused outright when the shards are not the same experiment: a different
    set of horizons, clock, support or conditions, or shard numbers that are
    not every index of one count. Merged either way when they are, with the
    exchangeability of each shard's anchors reported for the authority gate.
    """
    if not payloads:
        raise ValueError("no shard files to merge")
    counts = {int(str(p["shard"]).split("/")[1]) for p in payloads}
    if len(counts) != 1:
        raise ValueError(f"shards disagree about how many there are: {sorted(counts)}")
    count = counts.pop()
    indices = sorted(int(str(p["shard"]).split("/")[0]) for p in payloads)
    if indices != list(range(count)):
        raise ValueError(f"shards {indices} are not every index of {count}")
    for payload in payloads:
        if [int(lag) for lag in payload["lags"]] != [int(lag) for lag in lags]:
            raise ValueError(f"shard {payload['shard']} scored other horizons: {payload['lags']}")
        if abs(float(payload["frame_seconds"]) - float(frame_seconds)) > 1e-9:
            raise ValueError(f"shard {payload['shard']} ran on another clock: {payload['frame_seconds']}")
        if list(payload["support"]) != list(support) or list(payload["conditions"]) != list(conditions):
            raise ValueError(f"shard {payload['shard']} tested another support or other conditions")
    checks = {
        str(p["shard"]): anchors_exchangeable(
            reference_anchors, np.asarray(p["anchor_states"], dtype=np.float64),
            comparisons=count, alpha=alpha, seed=seed + index,
        )
        for index, p in enumerate(payloads)
    }
    reports = {
        int(lag): merge_sweeps(
            [p["reports"][f"lag_{int(lag)}"] for p in payloads], cuts_in_full=cuts_in_full
        )
        for lag in lags
    }
    gate = {
        "shards": count,
        "exchangeable": all(check.get("exchangeable") for check in checks.values()),
        "by_shard": checks,
        "why": (
            "every shard's anchors are exchangeable with the coordinator's"
            if all(check.get("exchangeable") for check in checks.values())
            else "; ".join(f"shard {name}: {check.get('why')}" for name, check in checks.items() if not check.get("exchangeable"))
        ),
    }
    return reports, gate
