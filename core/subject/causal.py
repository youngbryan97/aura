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
from core.subject.state import DOMAINS, CoreState, perturb

__all__ = [
    "EDGE_EFFECT",
    "power_note",
    "EDGE_QVALUE",
    "EDGE_REPLICATION",
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


@dataclass
class InterventionSet:
    """Every trial run, and the edges that survive them."""

    trials: list[Trial] = field(default_factory=list)
    scale: dict[str, np.ndarray] = field(default_factory=dict)
    delta: float = DEFAULT_DELTA
    lags: int = 0
    unwritable: tuple[str, ...] = ()
    seconds: float = 0.0

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
    """Per-domain distance between two runs, in units of ordinary variation."""
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
            series.append(float(np.sqrt(np.mean(gap**2))))
        trace[domain] = series
        peak[domain] = max(series) if series else 0.0
    return peak, trace


async def _arm(
    runtime: SubjectRuntime,
    snapshot: Any,
    condition: Condition,
    *,
    turns: int,
    displace: tuple[str, float] | None,
) -> list[CoreState]:
    runtime.restore(snapshot)
    frames: list[CoreState] = []
    applied = {"done": displace is None}

    def hit(rt: SubjectRuntime) -> None:
        if displace is None or applied["done"]:
            return
        domain, delta = displace
        applied["done"] = perturb(rt.state, domain, delta, ontogeny=rt.ontogeny)

    for turn in range(turns):
        frames.extend(
            await runtime.turn_once(
                condition,
                perturb_at=0 if (turn == 0 and displace is not None) else None,
                perturb=hit,
            )
        )
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
    on_progress: Callable[[str], None] | None = None,
) -> InterventionSet:
    """Displace each domain in each condition, repeatedly, against two shams."""
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
            snapshot = runtime.snapshot()
            for source in sources:
                if source in unwritable:
                    continue
                order = [("pert", (source, delta)), ("sham_a", None), ("sham_b", None)]
                rng.shuffle(order)
                runs: dict[str, list[CoreState]] = {}
                for name, displace in order:
                    runs[name] = await _arm(
                        runtime, snapshot, condition, turns=turns, displace=displace
                    )
                if not runs["pert"]:
                    unwritable.add(source)
                    logger.debug("%s has no writer that bit in %s", source, condition.name)
                    continue
                effect, trace = _divergence(runs["pert"], runs["sham_a"], scale)
                floor, floor_trace = _divergence(runs["sham_a"], runs["sham_b"], scale)
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
                    )
                )
                out.lags = max(out.lags, max((len(v) for v in trace.values()), default=0))
            runtime.restore(snapshot)
            if on_progress is not None:
                on_progress(f"trial {index + 1}/{trials} {condition.name}")

    out.unwritable = tuple(sorted(unwritable))
    out.seconds = _time.monotonic() - started
    return out


# ── from trials to edges ─────────────────────────────────────────────────


#: Sign-flip draws. The smallest p this can return is 1/(draws+1), so a run
#: with few paired trials cannot reach the q it is being judged against. That
#: is a property of the design, not of the system, and `build_edges` reports it
#: rather than returning an empty graph that looks like a negative result.
SIGN_FLIP_DRAWS: int = 4000


def _sign_flip_p(differences: np.ndarray, *, seed: int, draws: int = SIGN_FLIP_DRAWS) -> float:
    """One-sided paired test with no distributional assumption.

    The pairing is the point: within a trial the displaced arm and the two
    sham arms share a snapshot, so the difference between what the
    intervention moved and what nothing moved is the quantity with a null of
    zero. Flipping signs is the exact null for that difference.
    """
    if differences.size == 0:
        return 1.0
    observed = float(differences.mean())
    if observed <= 0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(draws, differences.size))
    null = (signs * differences).mean(axis=1)
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
    raw: list[tuple[str, str, float, float, tuple[str, ...], int]] = []

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
        for condition in sorted({trial.condition for trial in trials}):
            local = np.array(
                [
                    trial.effect[target] - trial.floor.get(target, 0.0)
                    for trial in trials
                    if trial.condition == condition
                ],
                dtype=np.float64,
            )
            if local.size and float(local.mean()) >= effect_min:
                replicated.append(condition)
        raw.append((source, target, effect, p, tuple(replicated), len(trials)))

    q_values = benjamini_hochberg([item[3] for item in raw])
    edges: list[Edge] = []
    for (source, target, effect, p, replicated, count), q in zip(raw, q_values, strict=True):
        record = {
            "source": source,
            "target": target,
            "effect": round(effect, 4),
            "p": round(p, 6),
            "q": round(q, 6),
            "replication": len(replicated),
            "trials": count,
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
