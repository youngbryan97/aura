"""core/interiority/longitudinal.py — the properties that only appear over time.

O5 on the council docket, and the class of claim the single-tick proofs
cannot reach. A counterfactual asks what one appraisal does. An ablation
asks whether one faculty matters. Neither can see whether a bond decays
the way a bond should, whether an anniversary still lands after a quiet
year, whether a channel that has been shouted at goes deaf, or whether
someone who started complying can get their standing back.

Those are the claims most likely to be wrong, because they are the ones
nobody exercises by hand. Each episode below runs the real machinery
over many steps and asserts a shape rather than a value: what must rise,
what must fall, what must not move, and what must be recoverable.

The shapes are the interesting part. A decay curve and an extinction
process both go down; only one of them comes back when you walk into the
old kitchen. A tolerance and a broken channel both go quiet; only one of
them recovers in silence. Testing the endpoint would pass either.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


@dataclass
class Episode:
    """One long-running property, and what its trace must look like."""

    name: str
    question: str
    run: Callable[[], dict[str, Any]]
    #: Reads the trace and returns (held, detail).
    check: Callable[[dict[str, Any]], tuple[bool, str]]


@dataclass
class EpisodeResult:
    name: str
    question: str
    held: bool
    detail: str
    trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": self.name,
            "question": self.question,
            "held": self.held,
            "detail": self.detail,
        }


# ── the episodes ──────────────────────────────────────────────────────
def _baseline_sharpens() -> dict[str, Any]:
    """Does a read get sharper as a person's baseline accumulates?

    Measured against a realistic pattern: mostly ordinary exchanges with an
    occasional departure. The first version alternated calm and agitated
    every other message, and confidence fell — correctly, because a person
    who alternates every other message has agitation *as* their baseline,
    and there is no deviation left to read. That is the baseline working,
    not failing, and the episode was asking the wrong question.
    """
    from core.interiority.event import EventKind, InteriorEvent
    from core.interiority.evidence import measured
    from core.interiority.other_minds import OtherMindsModel

    def ordinary() -> InteriorEvent:
        return InteriorEvent(
            kind=EventKind.SOCIAL, subject="p",
            observations={"timing": measured(0.22), "lexical": measured(0.18)},
        )

    def departure() -> InteriorEvent:
        return InteriorEvent(
            kind=EventKind.SOCIAL, subject="p",
            observations={"timing": measured(0.9), "lexical": measured(0.85)},
        )

    confidences: list[float] = []
    depths = (2, 6, 14, 30)
    for depth in depths:
        model = OtherMindsModel()
        for _ in range(depth):
            model.estimate(ordinary())
        confidences.append(model.estimate(departure()).confidence)
    return {"confidences": confidences, "baseline_depths": list(depths)}


def _check_baseline_sharpens(trace: dict[str, Any]) -> tuple[bool, str]:
    c = trace["confidences"]
    depths = trace["baseline_depths"]
    return (
        c[-1] >= c[0],
        "confidence reading one departure, against baselines of "
        + ", ".join(f"{d} exchanges: {v:.3f}" for d, v in zip(depths, c)),
    )


def _grief_returns_at_the_anniversary() -> dict[str, Any]:
    """Acute grief falls with time. The continuing bond must not."""
    from core.interiority.ledger import RelationalLedger

    ledger = RelationalLedger()
    ledger.bond("them", 0.9)
    loss = ledger.register_loss(
        "them", irreversibility=1.0,
        contexts=("the kitchen", "their birthday", "the walk"),
    )
    now = time.time()
    acute_day_one = loss.acute(now)
    acute_year_on = loss.acute(now + 365 * 86400)
    continuing_before = loss.continuing()
    # A year of not going anywhere near it.
    continuing_year_on = loss.continuing()
    # Then one context is actually met.
    gained = ledger.visit_context("them", "the kitchen")
    continuing_after_visit = ledger.loss_for("them").continuing()
    return {
        "acute_day_one": acute_day_one,
        "acute_year_on": acute_year_on,
        "continuing_before": continuing_before,
        "continuing_year_on": continuing_year_on,
        "continuing_after_visit": continuing_after_visit,
        "integration_gained": gained,
    }


def _check_grief_shape(trace: dict[str, Any]) -> tuple[bool, str]:
    fell = trace["acute_year_on"] < trace["acute_day_one"]
    time_alone_did_nothing = (
        abs(trace["continuing_year_on"] - trace["continuing_before"]) < 1e-9
    )
    contact_did_something = trace["continuing_after_visit"] < trace["continuing_before"]
    return (
        fell and time_alone_did_nothing and contact_did_something,
        (
            f"acute {trace['acute_day_one']:.3f} -> {trace['acute_year_on']:.3f} "
            f"over a year; continuing bond unchanged at "
            f"{trace['continuing_year_on']:.3f} until a context was met, then "
            f"{trace['continuing_after_visit']:.3f}"
        ),
    )


def _a_shouted_channel_goes_quiet_and_recovers() -> dict[str, Any]:
    from core.interiority.receptors import Receptor

    receptor = Receptor("probe")
    first = receptor.transduce(1.0, dt=0.0)
    for _ in range(30):
        receptor.transduce(1.0, dt=1.0)
    saturated = receptor.transduce(1.0, dt=1.0)
    withdrawal_now = receptor.withdrawal()
    for _ in range(600):
        receptor.transduce(0.0, dt=1.0)
    recovered = receptor.transduce(1.0, dt=0.0)
    return {
        "first": first,
        "after_thirty_seconds_of_it": saturated,
        "withdrawal_during": withdrawal_now,
        "after_ten_minutes_of_silence": recovered,
    }


def _check_tolerance_recovers(trace: dict[str, Any]) -> tuple[bool, str]:
    adapted = trace["after_thirty_seconds_of_it"] < trace["first"] * 0.5
    recovered = trace["after_ten_minutes_of_silence"] > trace["after_thirty_seconds_of_it"]
    return (
        adapted and recovered,
        (
            f"{trace['first']:.3f} -> {trace['after_thirty_seconds_of_it']:.3f} "
            f"under thirty seconds of it, back to "
            f"{trace['after_ten_minutes_of_silence']:.3f} after ten minutes of "
            "silence"
        ),
    )


def _an_obligation_outlasts_the_mood() -> dict[str, Any]:
    from core.interiority.ledger import RelationalLedger

    ledger = RelationalLedger()
    ledger.take_custody("c", "the_cat", vulnerability=0.9)
    still_held = []
    for _ in range(200):
        still_held.append(bool(ledger.custody_for("the_cat")))
    ledger.release_custody("c")
    after_release = bool(ledger.custody_for("the_cat"))
    return {"held_throughout": all(still_held), "after_release": after_release}


def _check_obligation_persists(trace: dict[str, Any]) -> tuple[bool, str]:
    return (
        trace["held_throughout"] and not trace["after_release"],
        (
            "custody held across two hundred steps and ended only on release: "
            f"{trace['held_throughout']}, released: {not trace['after_release']}"
        ),
    )


def _standing_can_be_recovered() -> dict[str, Any]:
    """Someone who starts complying must be able to get their standing back."""
    from core.interiority.ledger import RelationalLedger

    ledger = RelationalLedger()
    trace: list[float] = []

    def wtr() -> float:
        ignored = ledger.notes.times_seen("ignored_request", "them")
        heeded = ledger.notes.times_seen("heeded_request", "them")
        return (heeded + 1.0) / (heeded + ignored + 2.0)

    trace.append(wtr())
    for _ in range(3):
        ledger.notes.note_seen("ignored_request", "them")
        trace.append(wtr())
    lowest = trace[-1]
    for _ in range(12):
        ledger.notes.note_seen("heeded_request", "them")
        trace.append(wtr())
    return {"trace": trace, "lowest": lowest, "recovered": trace[-1]}


def _check_standing_recovers(trace: dict[str, Any]) -> tuple[bool, str]:
    fell = trace["lowest"] < 0.25
    came_back = trace["recovered"] > 0.5
    return (
        fell and came_back,
        (
            f"three ignored requests took the estimate to {trace['lowest']:.3f}, "
            f"below the threshold; twelve heeded ones brought it to "
            f"{trace['recovered']:.3f}"
        ),
    )


def _a_standard_can_stop_being_one_she_holds() -> dict[str, Any]:
    from core.interiority.ledger import RelationalLedger

    ledger = RelationalLedger()
    ledger.standing.norm("held", weight=0.9, endorsement=0.5)
    ledger.standing.norm("obeyed", weight=0.9, endorsement=0.5)
    held_trace, obeyed_trace = [], []
    for _ in range(40):
        ledger.standing.reinforce_norm("held", served_her_own=True)
        ledger.standing.reinforce_norm("obeyed", served_her_own=False)
        held_trace.append(ledger.standing.norm_for("held").endorsement)
        obeyed_trace.append(ledger.standing.norm_for("obeyed").endorsement)
    return {"held": held_trace, "obeyed": obeyed_trace}


def _check_endorsement_diverges(trace: dict[str, Any]) -> tuple[bool, str]:
    held, obeyed = trace["held"][-1], trace["obeyed"][-1]
    return (
        held > 0.75 and obeyed < 0.25,
        f"after forty outcomes: held {held:.3f}, merely obeyed {obeyed:.3f}",
    )


def _a_trace_expires() -> dict[str, Any]:
    """An outcome a day later must collect nothing."""
    from core.interiority.attribution import Attribution

    attribution = Attribution()
    trace = attribution.note_firing(
        "probe", event_id="e", intensity=0.9, claim="approach"
    )
    fresh = trace.eligibility(time.time())
    day_later = trace.eligibility(time.time() + 86400)
    return {"fresh": fresh, "a_day_later": day_later}


def _check_trace_expires(trace: dict[str, Any]) -> tuple[bool, str]:
    return (
        trace["fresh"] > 0.9 and trace["a_day_later"] < 0.01,
        (
            f"eligibility {trace['fresh']:.3f} now, "
            f"{trace['a_day_later']:.2e} a day later"
        ),
    )


EPISODES: tuple[Episode, ...] = (
    Episode(
        "baseline_sharpens_a_read",
        "does a read on a person get sharper as their baseline accumulates?",
        _baseline_sharpens, _check_baseline_sharpens,
    ),
    Episode(
        "grief_returns_at_the_anniversary",
        "does the continuing bond survive a year in which nothing was met?",
        _grief_returns_at_the_anniversary, _check_grief_shape,
    ),
    Episode(
        "a_shouted_channel_recovers",
        "does a channel that adapted away its gain get it back in silence?",
        _a_shouted_channel_goes_quiet_and_recovers, _check_tolerance_recovers,
    ),
    Episode(
        "an_obligation_outlasts_the_mood",
        "does custody hold across two hundred steps and end only on release?",
        _an_obligation_outlasts_the_mood, _check_obligation_persists,
    ),
    Episode(
        "standing_can_be_recovered",
        "can someone who starts complying get their standing back?",
        _standing_can_be_recovered, _check_standing_recovers,
    ),
    Episode(
        "a_standard_can_stop_being_held",
        "does a standard that never serves her drift toward merely obeyed?",
        _a_standard_can_stop_being_one_she_holds, _check_endorsement_diverges,
    ),
    Episode(
        "a_trace_expires",
        "does an outcome a day later collect nothing?",
        _a_trace_expires, _check_trace_expires,
    ),
)


def run_episodes(episodes: Sequence[Episode] | None = None) -> list[EpisodeResult]:
    results: list[EpisodeResult] = []
    for episode in episodes or EPISODES:
        trace = episode.run()
        held, detail = episode.check(trace)
        results.append(
            EpisodeResult(
                name=episode.name,
                question=episode.question,
                held=held,
                detail=detail,
                trace=trace,
            )
        )
    return results


def summary() -> dict[str, Any]:
    results = run_episodes()
    return {
        "episodes": len(results),
        "held": sum(1 for r in results if r.held),
        "results": [r.to_dict() for r in results],
        "failed": [r.to_dict() for r in results if not r.held],
    }


__all__ = ["EPISODES", "Episode", "EpisodeResult", "run_episodes", "summary"]
