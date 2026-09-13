"""Interventions: the only evidence this battery lets into the graph.

Every other kind of edge is available cheaply and none of them is asked for
here. An import is not an edge, a call is not an edge, and a correlation
between two channels is not an edge. An edge is

    P(X_j(t+tau) | do(X_i = x_i + delta)) != P(X_j(t+tau) | sham)

measured by running the same life twice from the same forked state and
comparing what the two runs did.

Three arms, not two. The first is displaced, the second is left alone, and the
third is also left alone. The third arm is what makes the number readable: two
untouched runs of the same organism from the same snapshot do not produce
identical trajectories — wall clocks move, a hash lands differently, a phase
takes a different branch on a millisecond — and without measuring that floor
directly every one of those differences would be counted as causal influence.
So the effect reported is the displaced run's divergence minus the divergence
of two runs of the same untouched life, in units of how much that domain
varies during ordinary operation.

The edge criteria were fixed before the first run: a false discovery rate
below 0.01 across all pairs tested, a standardised effect of at least 0.3, and
the same edge reaching those in at least three of the eight conditions. They
are engineering thresholds, not laws, and their only job is to be unchangeable
after the result is seen.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.subject.driver import Condition, SubjectRuntime
from core.subject.state import DOMAINS, CoreState, perturb, perturb_organs

__all__ = [
    "EDGE_EFFECT",
    "power_note",
    "EDGE_QVALUE",
    "EDGE_REPLICATION",
    "SUSTAINED",
    "Edge",
    "InterventionSet",
    "Trial",
    "benjamini_hochberg",
    "build_edges",
    "run_interventions",
]

logger = logging.getLogger("Aura.Subject.Causal")

#: Preregistered. Written here, once, and read by the battery; changing one is
#: a diff on this line rather than a different call site.
EDGE_QVALUE: float = 0.01
EDGE_EFFECT: float = 0.3
EDGE_REPLICATION: int = 3

#: Displacement size. Large enough that a domain's own reading moves by more
#: than its ordinary within-turn wobble, small enough to leave the state in
#: the operating range every writer clamps to.
DEFAULT_DELTA: float = 0.15

#: Domains whose displacement is held for the arm rather than delivered once.
#:
#: `do(X)` holds X at the value it was set to; a single push is not that. It
#: makes no difference to a domain whose state persists — a goal stays on the
#: list, a budget stays where it was filled to — and all the difference to one
#: whose own dynamics pull it back faster than its consumers look at it.
#:
#: Recurrent cognition is the case. The substrate integrates at twenty hertz
#: with a time constant of a tenth of a second, and the homeostatic coupling
#: that carries it into how hot and how deep she may think runs on the
#: heartbeat, once a second. So a push delivered early in a turn had decayed by
#: five e-foldings before anything downstream looked, and C measured as a
#: domain that can be moved ten standard deviations and reach a tenth of one
#: anywhere else. Held, the consumers see the displacement whenever they
#: sample.
#:
#: Membership follows from the layer rates the campaign already freezes, not
#: from which domains came out badly: a domain backed by a layer whose period
#: is shorter than a turn cannot deliver a pulse to a once-a-turn consumer.
SUSTAINED: frozenset[str] = frozenset({"C"})

#: A column with less spread than this during ordinary operation has no scale
#: to be measured against and is left out of every distance.
SCALE_FLOOR: float = 1e-6

#: Standardised differences are clipped here before a domain's columns are
#: combined. Some columns barely move during ordinary work and move a great
#: deal under intervention; without a ceiling one of them turns a domain
#: distance into twenty thousand standard deviations, which is not a stronger
#: finding than ten, only a less readable one.
DIVERGENCE_CEILING: float = 10.0


@dataclass(frozen=True, slots=True)
class Trial:
    """One paired intervention: what moved, and what moved without a cause."""

    source: str
    condition: str
    index: int
    #: target domain -> divergence of the displaced arm from the sham arm
    effect: dict[str, float]
    #: target domain -> divergence of two sham arms from each other
    floor: dict[str, float]
    #: per-lag divergence, for the perturbational measures
    trace: dict[str, list[float]]
    floor_trace: dict[str, list[float]]
    took: bool
    #: How far the displaced domain itself moved. The ratio of this to what it
    #: moved in other domains is the coupling gain, and a system whose parts
    #: exchange compressed summaries shows a large one.
    self_effect: float = 0.0
    injected_at: int = 0


@dataclass
class InterventionSet:
    """Every trial run, and the edges that survive them."""

    trials: list[Trial] = field(default_factory=list)
    scale: dict[str, np.ndarray] = field(default_factory=dict)
    delta: float = DEFAULT_DELTA
    lags: int = 0
    unwritable: tuple[str, ...] = ()
    seconds: float = 0.0

    def floor_report(self) -> dict[str, Any]:
        """What two untouched arms did, broken out every way it can be read.

        Sham against sham is the instrument's own noise, and an effect bar it
        approaches is a bar that is measuring restoration rather than coupling.
        One pooled number cannot say that: a floor that is clean in seven
        domains and half the threshold in the eighth leaves exactly one domain
        uninterpretable, and the pooled mean hides which.

        So it is reported by target domain, by condition, by source, and over
        lag, with the share of the effect threshold each one takes. Any domain
        whose floor reaches a third of the bar is named: a negative result
        there is not interpretable, whatever the pooled number says.
        """
        if not self.trials:
            return {"trials": 0}

        def _summary(values: list[float]) -> dict[str, float]:
            array = np.asarray(values, dtype=np.float64)
            return {
                "mean": round(float(array.mean()), 5),
                "q95": round(float(np.quantile(array, 0.95)), 5),
                "max": round(float(array.max()), 5),
            }

        by_target: dict[str, dict[str, float]] = {}
        by_condition: dict[str, dict[str, float]] = {}
        by_source: dict[str, dict[str, float]] = {}
        for target in DOMAINS:
            values = [
                t.floor.get(target, 0.0) for t in self.trials if target in t.floor
            ]
            if values:
                by_target[target] = _summary(values)
        for condition in sorted({t.condition for t in self.trials}):
            values = [
                value
                for t in self.trials
                if t.condition == condition
                for value in t.floor.values()
            ]
            if values:
                by_condition[condition] = _summary(values)
        for source in sorted({t.source for t in self.trials}):
            values = [
                value for t in self.trials if t.source == source
                for value in t.floor.values()
            ]
            if values:
                by_source[source] = _summary(values)

        # Over lag, pooled across every trial and target: restoration noise
        # that grows with the horizon is a different problem from restoration
        # noise that is there on the first frame.
        by_lag: list[float] = []
        width = max(
            (len(trace) for t in self.trials for trace in t.floor_trace.values()),
            default=0,
        )
        for lag in range(width):
            values = [
                trace[lag]
                for t in self.trials
                for trace in t.floor_trace.values()
                if len(trace) > lag
            ]
            if values:
                by_lag.append(round(float(np.mean(values)), 5))

        crowded = sorted(
            target
            for target, row in by_target.items()
            if row["q95"] >= EDGE_EFFECT / 3.0
        )
        return {
            "trials": len(self.trials),
            "threshold": EDGE_EFFECT,
            "by_target": by_target,
            "by_condition": by_condition,
            "by_source": by_source,
            "by_lag": by_lag,
            "worst_target": max(by_target, key=lambda k: by_target[k]["q95"], default=None)
            if by_target
            else None,
            # A domain whose floor reaches a third of the bar. A negative
            # result for one of these is not interpretable.
            "crowded_by_restoration_noise": crowded,
            "floor_is_clean": not crowded,
        }

    def attenuation(self) -> dict[str, dict[str, float]]:
        """For each displaced domain: how much it moved, and how much got out.

        A large ratio is the shape of brokered or summarised coupling. It is
        reported whether or not any edge passed, because "no edge" and "an edge
        attenuated fifty times on the way out" are different findings and the
        graph alone cannot tell them apart.
        """
        out: dict[str, dict[str, float]] = {}
        for source in {trial.source for trial in self.trials}:
            mine = [t for t in self.trials if t.source == source]
            own = float(np.mean([t.self_effect for t in mine])) if mine else 0.0
            escaped = 0.0
            for target in DOMAINS:
                if target == source:
                    continue
                values = [
                    t.effect.get(target, 0.0) - t.floor.get(target, 0.0) for t in mine
                ]
                if values:
                    escaped = max(escaped, float(np.mean(values)))
            out[source] = {
                "self_effect": round(own, 4),
                "largest_outgoing": round(escaped, 4),
                "attenuation": round(own / escaped, 2) if escaped > 1e-9 else float("inf"),
            }
        return out

    def by_pair(self) -> dict[tuple[str, str], list[Trial]]:
        out: dict[tuple[str, str], list[Trial]] = {}
        for trial in self.trials:
            for target in trial.effect:
                out.setdefault((trial.source, target), []).append(trial)
        return out


@dataclass(frozen=True, slots=True)
class Edge:
    """One surviving i -> j, with what it survived."""

    source: str
    target: str
    effect: float
    p_value: float
    q_value: float
    conditions: tuple[str, ...]
    trials: int

    @property
    def replicated(self) -> int:
        return len(self.conditions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "effect": round(self.effect, 4),
            "p": round(self.p_value, 6),
            "q": round(self.q_value, 6),
            "conditions": list(self.conditions),
            "replication": self.replicated,
            "trials": self.trials,
        }


# ── running the arms ─────────────────────────────────────────────────────


def _divergence(
    left: Sequence[CoreState], right: Sequence[CoreState], scale: dict[str, np.ndarray]
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Per-domain distance between two runs, in units of ordinary variation.

    The distance is the largest standardized displacement among the domain's
    live columns, not the root mean square across them. The question an edge
    answers is whether displacing i changes j at all, and a mean over j's
    columns makes that answer depend on how finely j was written down: split
    the self-state into fifty columns instead of twenty-five and every
    single-channel effect halves, though nothing about the organism changed.
    An existence claim must not be a function of the schema's granularity.

    The floor is computed the same way from two sham arms. See
    `_paired_divergence` for why the two must be maximised over the same
    column rather than over the domain independently.
    """
    peak: dict[str, float] = {}
    trace: dict[str, list[float]] = {}
    span = min(len(left), len(right))
    for domain in DOMAINS:
        unit = scale.get(domain)
        if unit is None:
            continue
        live = unit > SCALE_FLOOR
        if not live.any():
            continue
        series: list[float] = []
        for index in range(span):
            gap = (left[index].domain(domain) - right[index].domain(domain))[live] / unit[live]
            gap = np.clip(np.abs(gap), 0.0, DIVERGENCE_CEILING)
            series.append(float(gap.max()) if gap.size else 0.0)
        trace[domain] = series
        peak[domain] = max(series) if series else 0.0
    return peak, trace


def _paired_divergence(
    pert: Sequence[CoreState],
    sham_a: Sequence[CoreState],
    sham_b: Sequence[CoreState],
    scale: dict[str, np.ndarray],
) -> tuple[dict[str, float], dict[str, float], dict[str, list[float]], dict[str, list[float]]]:
    """The effect and the floor, read off the same column.

    Both were a maximum over the domain's live columns, taken independently —
    the largest effect anywhere against the largest sham wobble anywhere. When
    the two maxima land on different columns the subtraction the edge rule
    makes is not a comparison at all, and it is biased in one direction: the
    effect has to beat a floor set by a column it never touched.

    It bites hardest on columns that barely move. Recurrent cognition's
    frustration channel has a standard deviation of 0.0019 across a recording,
    the fourth smallest of the live columns, so an absolute difference of three
    ten-thousandths between two identical arms reads as a seventh of a standard
    deviation — and that was the number every effect into that domain had to
    clear, whichever column carried it.

    So the column is chosen once, by the margin, and both numbers are read off
    it. The choice is still a maximum over columns, which is what keeps the
    answer from depending on how finely the domain was written down; what it no
    longer does is let one column's noise stand in front of another column's
    signal.
    """
    effect: dict[str, float] = {}
    floor: dict[str, float] = {}
    trace: dict[str, list[float]] = {}
    floor_trace: dict[str, list[float]] = {}
    span = min(len(pert), len(sham_a), len(sham_b))
    for domain in DOMAINS:
        unit = scale.get(domain)
        if unit is None:
            continue
        live = unit > SCALE_FLOOR
        if not live.any() or span == 0:
            continue
        scaled = unit[live]
        moved = np.zeros((span, int(live.sum())), dtype=np.float64)
        wobble = np.zeros_like(moved)
        for index in range(span):
            here = pert[index].domain(domain)[live]
            there = sham_a[index].domain(domain)[live]
            other = sham_b[index].domain(domain)[live]
            moved[index] = np.clip(np.abs(here - there) / scaled, 0.0, DIVERGENCE_CEILING)
            wobble[index] = np.clip(np.abs(there - other) / scaled, 0.0, DIVERGENCE_CEILING)
        # Each column's own peak over the lags, then the column whose margin
        # over its own floor is largest.
        column_effect = moved.max(axis=0)
        column_floor = wobble.max(axis=0)
        best = int(np.argmax(column_effect - column_floor))
        effect[domain] = float(column_effect[best])
        floor[domain] = float(column_floor[best])
        trace[domain] = [float(value) for value in moved[:, best]]
        floor_trace[domain] = [float(value) for value in wobble[:, best]]
    return effect, floor, trace, floor_trace


async def _arm(
    runtime: SubjectRuntime,
    snapshot: Any,
    condition: Condition,
    *,
    turns: int,
    displace: tuple[str, float] | None,
    at: int = 0,
) -> list[CoreState]:
    runtime.restore(snapshot)
    # The freeze this arm was handed, kept so the arm can hand it back. An
    # interoceptive displacement re-takes the freeze from the displaced state,
    # which is what a sustained perturbation of the body is — and the runtime
    # is one object shared by all three arms, so without this the re-freeze
    # survived into the shams. With the arm order shuffled, two trials in three
    # ran a sham under the displaced body: the effect on I read as zero because
    # both arms ended at the same body, and every downstream consequence of the
    # displacement was present in the floor as well, cancelling itself.
    held_host = None if runtime.frozen_host is None else dict(runtime.frozen_host)
    held_latency = None if runtime.frozen_latency is None else dict(runtime.frozen_latency)
    frames: list[CoreState] = []
    applied = {"done": displace is None}
    sustained = displace is not None and displace[0] in SUSTAINED

    async def hit(rt: SubjectRuntime) -> None:
        """Write to both halves of the domain: the state fields and the organ."""
        if displace is None or (applied["done"] and not sustained):
            return
        domain, delta = displace
        in_state = perturb(rt.state, domain, delta, ontogeny=rt.ontogeny)
        in_organ = await perturb_organs(rt.organs, domain, delta, state=rt.state)
        if domain == "I" and rt.frozen_host is not None:
            # The body readings are held still for the duration of a trial so
            # the host's own load cannot drift between arms. That hold also
            # erased the one perturbation aimed at the body, which is why
            # displacing interoception reached nothing at all. Re-taking the
            # freeze from the displaced state carries the displacement instead
            # of overwriting it — which is what a sustained interoceptive
            # perturbation is.
            rt.freeze_host()
        applied["done"] = applied["done"] or bool(in_state or in_organ)

    try:
        for turn in range(turns):
            frames.extend(
                await runtime.turn_once(
                    condition,
                    perturb_at=(
                        at
                        if (turn == 0 and displace is not None)
                        else (0 if sustained else None)
                    ),
                    perturb=hit,
                    sustain=hit if sustained else None,
                )
            )
    finally:
        runtime.frozen_host = held_host
        runtime.frozen_latency = held_latency
    if displace is not None and not applied["done"]:
        return []
    return frames


async def run_interventions(
    runtime: SubjectRuntime,
    conditions: Sequence[Condition],
    *,
    scale: dict[str, np.ndarray],
    sources: Sequence[str] = DOMAINS,
    trials: int = 6,
    turns: int = 2,
    delta: float = DEFAULT_DELTA,
    seed: int = 0,
    warmup: int = 4,
    injection_points: Sequence[int] = (0, 8, 16),
    on_progress: Callable[[str], None] | None = None,
) -> InterventionSet:
    """Displace each domain in each condition, repeatedly, against two shams.

    The displacement is injected at a different point of the turn on different
    trials. A phase that recomputes a domain from its own sources erases a
    displacement that arrived before it, and injecting only at the top of the
    turn would measure that erasure and report it as an absent edge. Cycling
    the injection point is the same move as stimulating at different depths.
    """
    import time as _time

    started = _time.monotonic()
    rng = random.Random(seed)
    out = InterventionSet(scale=scale, delta=delta)
    unwritable: set[str] = set()

    for _ in range(warmup):
        await runtime.turn_once(conditions[0])

    for index in range(trials):
        for condition in conditions:
            # Let the life move on between trials so the snapshots are taken at
            # different points of the trajectory rather than at one lucky spot.
            await runtime.turn_once(condition)
            # Hold the machine still for the three arms. What the body senses
            # is E, not K, and letting it drift between arms puts the host's
            # own load into the sham floor — measured at ten units of
            # temperature, which is larger than any effect this looks for.
            runtime.freeze_host()
            snapshot = runtime.snapshot()
            for source in sources:
                if source in unwritable:
                    continue
                where = injection_points[index % len(injection_points)]
                order = [("pert", (source, delta)), ("sham_a", None), ("sham_b", None)]
                rng.shuffle(order)
                runs: dict[str, list[CoreState]] = {}
                for name, displace in order:
                    runs[name] = await _arm(
                        runtime, snapshot, condition, turns=turns, displace=displace, at=where
                    )
                if not runs["pert"]:
                    unwritable.add(source)
                    logger.debug("%s has no writer that bit in %s", source, condition.name)
                    continue
                effect, floor, trace, floor_trace = _paired_divergence(
                    runs["pert"], runs["sham_a"], runs["sham_b"], scale
                )
                out.trials.append(
                    Trial(
                        source=source,
                        condition=condition.name,
                        index=index,
                        effect=effect,
                        floor=floor,
                        trace=trace,
                        floor_trace=floor_trace,
                        took=True,
                        self_effect=effect.get(source, 0.0) - floor.get(source, 0.0),
                        injected_at=where,
                    )
                )
                out.lags = max(out.lags, max((len(v) for v in trace.values()), default=0))
            runtime.restore(snapshot)
            runtime.thaw_host()
            if on_progress is not None:
                on_progress(f"trial {index + 1}/{trials} {condition.name}")

    out.unwritable = tuple(sorted(unwritable))
    out.seconds = _time.monotonic() - started
    return out


# ── from trials to edges ─────────────────────────────────────────────────


#: Sign-flip draws. The smallest p this can return is 1/(draws+1), and after
#: multiplicity correction across ninety pairs the smallest reachable q is
#: ninety times that. At four thousand draws — where this started — no edge
#: could reach q < 0.01 however real it was, and the run reported an empty
#: graph that read as a negative result about the organism. It was a fact about
#: the number of draws.
SIGN_FLIP_DRAWS: int = 200_000


def _sign_flip_p(differences: np.ndarray, *, seed: int, draws: int = SIGN_FLIP_DRAWS) -> float:
    """One-sided paired test with no distributional assumption.

    The pairing is the point: within a trial the displaced arm and the two sham
    arms share a snapshot, so the difference between what the intervention
    moved and what nothing moved is the quantity with a null of zero. Flipping
    signs is the exact null for that difference.

    The statistic is the signed rank, not the mean. On the mean, a single trial
    where the displacement happened to land hard carries the whole test, and
    the null built by flipping signs has a fat tail for the same reason — so a
    pair that moved in seven of eight conditions with an effect of seven tenths
    came out at q = 0.016 against a bar of 0.01, on evidence that is not
    marginal at all. The signed rank asks how consistently the difference is
    positive rather than how large its average is, which is the question the
    replication bar asks beside it, and it cuts both ways: an effect that is
    one enormous trial and forty-seven nothings no longer certifies.

    Exact zeros are dropped rather than ranked. A difference of zero is
    evidence of nothing in either direction, and giving it a rank lets the
    count of ties decide the answer.
    """
    if differences.size == 0:
        return 1.0
    kept = differences[np.abs(differences) > 1e-12]
    if kept.size == 0 or float(kept.mean()) <= 0.0:
        return 1.0
    order = np.argsort(np.abs(kept), kind="stable")
    ranks = np.empty(kept.size, dtype=np.float64)
    ranks[order] = np.arange(1, kept.size + 1, dtype=np.float64)
    signed = np.sign(kept) * ranks
    observed = float(signed.sum())
    if observed <= 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    flips = rng.choice((-1.0, 1.0), size=(draws, kept.size))
    null = (flips * ranks).sum(axis=1)
    return float((1.0 + np.sum(null >= observed)) / (draws + 1.0))


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """q-values, monotone, for the whole family of pairs tested at once."""
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(np.asarray(p_values, dtype=np.float64))
    ranked = np.asarray(p_values, dtype=np.float64)[order]
    q = ranked * n / (np.arange(n) + 1.0)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(q, 0.0, 1.0)
    return [float(v) for v in out]


def power_note(results: InterventionSet, *, q_max: float = EDGE_QVALUE) -> dict[str, Any]:
    """Whether this many trials could have found an edge at all.

    The smallest achievable p times the number of pairs tested is the smallest
    achievable q. If that is above the bar, no edge can pass however real it
    is, and reporting the empty graph without saying so would be reporting the
    sample size as a fact about the organism.
    """
    pairs = sum(1 for (a, b) in results.by_pair() if a != b)
    per_pair = min((len(v) for v in results.by_pair().values()), default=0)
    floor_p = 1.0 / (SIGN_FLIP_DRAWS + 1.0)
    smallest_q = floor_p * max(pairs, 1)
    return {
        "pairs_tested": pairs,
        "paired_trials_per_pair": per_pair,
        "smallest_reachable_q": round(smallest_q, 6),
        "underpowered": smallest_q > q_max or per_pair < 8,
    }


def build_edges(
    results: InterventionSet,
    *,
    seed: int = 0,
    q_max: float = EDGE_QVALUE,
    effect_min: float = EDGE_EFFECT,
    replication_min: int = EDGE_REPLICATION,
) -> tuple[list[Edge], list[dict[str, Any]]]:
    """Every pair tested, and the subset that met all three preregistered bars.

    A self-pair is dropped rather than scored. Displacing a domain moves that
    domain by construction, and counting it would put a loop on every node
    before any dynamics were consulted.
    """
    pairs = results.by_pair()
    tested: list[dict[str, Any]] = []
    raw: list[
        tuple[str, str, float, float, tuple[str, ...], int, dict[str, float], float, int]
    ] = []

    for (source, target), trials in sorted(pairs.items()):
        if source == target:
            continue
        differences = np.array(
            [trial.effect[target] - trial.floor.get(target, 0.0) for trial in trials],
            dtype=np.float64,
        )
        effect = float(differences.mean()) if differences.size else 0.0
        p = _sign_flip_p(differences, seed=abs(hash((source, target, seed))) % (2**31))
        replicated: list[str] = []
        by_condition: dict[str, float] = {}
        for condition in sorted({trial.condition for trial in trials}):
            local = np.array(
                [
                    trial.effect[target] - trial.floor.get(target, 0.0)
                    for trial in trials
                    if trial.condition == condition
                ],
                dtype=np.float64,
            )
            if not local.size:
                continue
            by_condition[condition] = round(float(local.mean()), 4)
            if float(local.mean()) >= effect_min:
                replicated.append(condition)
        # Where in the arm the response peaked, in frames after the
        # displacement. An effect that always peaks at the last frame recorded
        # is an effect the horizon cut off, and a horizon that is binding is a
        # fact about the measurement rather than about the organism.
        peaks = []
        for trial in trials:
            series = trial.trace.get(target) or []
            if series:
                peaks.append(int(np.argmax(series)))
        peak_lag = float(np.median(peaks)) if peaks else 0.0
        lags = max((len(trial.trace.get(target) or []) for trial in trials), default=0)
        raw.append(
            (source, target, effect, p, tuple(replicated), len(trials), by_condition, peak_lag, lags)
        )

    q_values = benjamini_hochberg([item[3] for item in raw])
    edges: list[Edge] = []
    for (
        source,
        target,
        effect,
        p,
        replicated,
        count,
        by_condition,
        peak_lag,
        lags,
    ), q in zip(raw, q_values, strict=True):
        # Which conditions carried it, not only how many. A pair that misses
        # the replication bar in one condition and a pair that carries in none
        # both read as "not kept" from the count alone, and they are the two
        # ends of the evidence.
        record = {
            "source": source,
            "target": target,
            "effect": round(effect, 4),
            "p": round(p, 6),
            "q": round(q, 6),
            "replication": len(replicated),
            "conditions": list(replicated),
            "by_condition": by_condition,
            "trials": count,
            "peak_lag": peak_lag,
            "lags": lags,
            "at_the_horizon": bool(lags and peak_lag >= lags - 1),
            "kept": False,
        }
        if q < q_max and effect >= effect_min and len(replicated) >= replication_min:
            record["kept"] = True
            edges.append(
                Edge(
                    source=source,
                    target=target,
                    effect=effect,
                    p_value=p,
                    q_value=q,
                    conditions=replicated,
                    trials=count,
                )
            )
        tested.append(record)
    return edges, tested
