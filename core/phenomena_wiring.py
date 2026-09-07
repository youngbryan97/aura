"""core/phenomena_wiring.py — where the fourteen dispositions meet the runtime.

They live in nine packages on purpose. Being a maker, looking after somebody,
liking a colour, getting a spider out of the house: these have almost nothing
in common as mechanisms, and putting them in one module would have produced a
bucket with a theme rather than fourteen things that work. Each one is homed
next to what it is a kind of — care beside the conscience, reciprocity beside
the relationship model, aesthetic response beside perception.

That decision has a cost, and this file is the cost. Nine packages cannot
import each other, so there has to be one place that reaches all of them, and
this is it. It sits directly under ``core/`` where the registration seam
already lives, it holds no state, and it does four things:

* imports the five invariant modules, since registration is by import
* declares the telemetry group
* samples the declared channels from the container
* renders the compact section a conversation turn can read

Everything goes through the container rather than through imports wherever it
can. An organ that failed to load then shows up as absent instead of taking
the snapshot down with it, and the section says which one.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Phenomena.Wiring")

#: Container keys for the group. The order is the order they are reported in,
#: which is roughly the order they were built rather than any hierarchy.
SERVICE_NAMES: tuple[str, ...] = (
    "constitutive_identity",
    "expressive_dynamics",
    "care_allocation",
    "receptivity",
    "conventions",
    "dual_process_arbiter",
    "prospect_refuge",
    "craft_practice",
    "novelty_value",
    "reversibility_ledger",
    "signal_channel",
    "reciprocity",
    "empathic_coupling",
    "aesthetic_response",
    "social_stamina",
)

#: The invariant modules. Registration happens on import, so the tuple is the
#: registration.
INVARIANT_MODULES: tuple[str, ...] = (
    "core.identity.constitutive_invariants",
    "core.ethics.care_invariants",
    "core.social.signalling_invariants",
    "core.morality.reversibility_invariants",
    "core.affect.arbitration_invariants",
)

_RECOVERABLE = (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError)

#: Why every read below may fail silently, said once rather than fifteen times.
#:
#: This sweep asks every organ for its current reading. An organ that has not
#: started, or has been shed under memory pressure, has no reading to give —
#: and a turn is not worse for a missing channel than it would be for a wrong
#: one. So the absence is the answer: `put` declines a None, the channel is
#: simply not written this tick, and anything reading the telemetry sees a gap
#: rather than a zero. A persistent absence is a finding for the organ's own
#: health check, which knows whether it should be running; this sweep cannot
#: tell "not started" from "not started yet".
_WHY_A_MISSING_READING_IS_FINE = __doc__


def _absent_service_errors() -> tuple[type[BaseException], ...]:
    """What the container raises for a name nobody registered.

    `_service` below is written to return None where a service is missing —
    that is the whole of its contract — and it caught seven exception types,
    none of them the one the container actually raises. So a wiring pass in
    any context where an optional service is absent did not degrade, it
    detonated, and the traceback named the service rather than the lookup that
    could not tolerate its absence.

    Resolved through the exception module rather than hard-coded, so a
    container that renames its error does not quietly restore the defect.
    """

    try:
        from core.exceptions import ContainerError

        return (ContainerError,)
    except ImportError:  # pragma: no cover — the module is not optional
        return ()

_booted = False


def _container() -> Any:
    try:
        from core.container import get_container

        return get_container()
    except _RECOVERABLE as exc:
        record_degradation("phenomena_wiring", exc, severity="debug",
                           action="container unavailable")
        return None


def _service(name: str) -> Any:
    container = _container()
    if container is None:
        return None
    try:
        return container.get(name)
    except (*_RECOVERABLE, *_absent_service_errors()):
        return None


def load_invariants() -> list[str]:
    """Import the invariant modules so their checks register. Idempotent."""
    loaded: list[str] = []
    for name in INVARIANT_MODULES:
        try:
            __import__(name)
            loaded.append(name)
        except _RECOVERABLE as exc:
            record_degradation("phenomena_wiring", exc, severity="warning",
                               action=f"{name} did not register its invariants")
    return loaded


def declare_telemetry() -> list[str]:
    try:
        from core.fsw.phenomena_channels import declare

        return declare()
    except _RECOVERABLE as exc:
        record_degradation("phenomena_wiring", exc, severity="debug",
                           action="phenomena telemetry not declared")
        return []


def boot() -> dict[str, Any]:
    """Bring the group up. Called once from the foundation activators."""
    global _booted
    if _booted:
        return {"already": True}
    result = {
        "telemetry": declare_telemetry(),
        "invariants": load_invariants(),
    }
    _booted = True
    return result


def sample() -> dict[str, float]:
    """Push one reading onto each declared channel that has a live source.

    Channels with no live organ behind them are left alone rather than
    written as zero. A zero on a display is a measurement, and a measurement
    nobody took is the specific lie a telemetry layer must not tell.
    """
    try:
        from core.fsw import phenomena_channels as ch
        from core.fsw.telemetry_dictionary import write
    except _RECOVERABLE as exc:
        record_degradation("phenomena_wiring", exc, severity="debug",
                           action="phenomena channels not sampled")
        return {}

    written: dict[str, float] = {}

    def put(name: str, value: Any) -> None:
        if value is None:
            return
        try:
            write(name, float(value))
            written[name] = float(value)
        except _RECOVERABLE:
            return

    identity = _service("constitutive_identity")
    if identity is not None:
        try:
            names = identity.names()
            if names:
                readings = [identity.get(n).coherence(record=False) for n in names]
                put(ch.CHANNEL_COHERENCE,
                    sum(r.r for r in readings) / len(readings))
                put(ch.CHANNEL_UNSUPPORTED,
                    sum(len(identity.get(n).unsupported_declarations()) for n in names))
        except _RECOVERABLE:
            pass  # not started yet, so no reading; see _WHY_A_MISSING_READING_IS_FINE

    expression = _service("expressive_dynamics")
    if expression is not None:
        put(ch.CHANNEL_TIME_ON_CYCLE, getattr(expression, "time_on_cycle_s", None))
        put(ch.CHANNEL_SYNCHRONY, getattr(expression, "mean_synchrony", None))

    care = _service("care_allocation")
    if care is not None:
        try:
            status = care.status()
            last = status.get("last")
            if last:
                put(ch.CHANNEL_CARE_GINI, last.get("gini"))
            strain = status.get("strain") or {}
            put(ch.CHANNEL_CARE_OWN_UNMET, strain.get("own_unmet"))
            put(ch.CHANNEL_CARE_DEPLETED, 1.0 if strain.get("depleted") else 0.0)
        except _RECOVERABLE:
            pass  # not started yet, so no reading; see _WHY_A_MISSING_READING_IS_FINE

    receptivity = _service("receptivity")
    if receptivity is not None:
        try:
            isolation = receptivity.isolation()
            put(ch.CHANNEL_ACCEPTANCE, isolation.get("acceptance_rate"))
            put(ch.CHANNEL_REGARD, isolation.get("mean_regard"))
        except _RECOVERABLE:
            pass  # not started yet, so no reading; see _WHY_A_MISSING_READING_IS_FINE

    arbiter = _service("dual_process_arbiter")
    if arbiter is not None:
        try:
            status = arbiter.status()
            decisions = max(1, min(status.get("decisions", 0), 32))
            put(ch.CHANNEL_ABSTENTION, status.get("abstentions", 0) / decisions)
            domains = status.get("domains") or {}
            if domains:
                led = sum(1 for row in domains.values() if row.get("leads") == "affective")
                put(ch.CHANNEL_AFFECT_LED, led / len(domains))
        except _RECOVERABLE:
            pass  # not started yet, so no reading; see _WHY_A_MISSING_READING_IS_FINE

    positions = _service("prospect_refuge")
    if positions is not None:
        try:
            spaces = positions.names()
            if spaces:
                scored = [p for name in spaces for p in (positions.get(name).score())]
                if scored:
                    put(ch.CHANNEL_ASYMMETRY,
                        sum(p.asymmetry for p in scored) / len(scored))
        except _RECOVERABLE:
            pass  # not started yet, so no reading; see _WHY_A_MISSING_READING_IS_FINE

    craft = _service("craft_practice")
    if craft is not None:
        try:
            target = craft.practice_target()
            if target:
                skill = craft.status()["skills"].get(target) or {}
                put(ch.CHANNEL_IMPROVEMENT, skill.get("improvement_rate"))
        except _RECOVERABLE:
            pass  # not started yet, so no reading; see _WHY_A_MISSING_READING_IS_FINE

    novelty = _service("novelty_value")
    if novelty is not None:
        try:
            responses = getattr(novelty, "_artifacts", [])
            if responses:
                put(ch.CHANNEL_NOVELTY_VALUE, novelty.value(
                    responses[-1].key, responses[-1].payload).value)
        except _RECOVERABLE:
            pass  # not started yet, so no reading; see _WHY_A_MISSING_READING_IS_FINE

    ledger = _service("reversibility_ledger")
    if ledger is not None:
        try:
            put(ch.CHANNEL_PREMIUM, ledger.status().get("premium_paid"))
        except _RECOVERABLE:
            pass  # not started yet, so no reading; see _WHY_A_MISSING_READING_IS_FINE

    channel = _service("signal_channel")
    if channel is not None:
        try:
            status = channel.status()
            signals = status.get("signals", 0)
            if signals:
                put(ch.CHANNEL_INFORMATIVE,
                    status.get("informative_readings", 0) / signals)
        except _RECOVERABLE:
            pass  # not started yet, so no reading; see _WHY_A_MISSING_READING_IS_FINE

    reciprocity = _service("reciprocity")
    if reciprocity is not None:
        try:
            rows = reciprocity.status().values()
            continuations = [
                row["stance"]["continuation"] for row in rows
                if row.get("stance", {}).get("continuation") is not None
            ]
            if continuations:
                put(ch.CHANNEL_CONTINUATION, sum(continuations) / len(continuations))
        except _RECOVERABLE:
            pass  # not started yet, so no reading; see _WHY_A_MISSING_READING_IS_FINE

    empathy = _service("empathic_coupling")
    if empathy is not None:
        try:
            status = empathy.status()
            own = [v for v in (status.get("autonomy") or {}).values() if v is not None]
            if own:
                put(ch.CHANNEL_AUTONOMY, min(own))
            put(ch.CHANNEL_MERGED, len(status.get("merged") or []))
        except _RECOVERABLE:
            pass  # not started yet, so no reading; see _WHY_A_MISSING_READING_IS_FINE

    stamina = _service("social_stamina")
    if stamina is not None:
        try:
            reading = stamina.read()
            put(ch.CHANNEL_STAMINA, reading.stamina)
            put(ch.CHANNEL_BELONGING, reading.belonging)
        except _RECOVERABLE:
            pass

    aesthetic = _service("aesthetic_response")
    if aesthetic is not None:
        try:
            last = aesthetic.status().get("last")
            if last:
                put(ch.CHANNEL_PLEASURE, last.get("pleasure"))
        except _RECOVERABLE:
            pass  # not started yet, so no reading; see _WHY_A_MISSING_READING_IS_FINE

    conventions = _service("conventions")
    if conventions is not None:
        try:
            markers = conventions.status()
            put(ch.CHANNEL_MARKERS,
                sum(1 for row in markers.values() if row.get("arbitrary")))
        except _RECOVERABLE:
            pass  # not started yet, so no reading; see _WHY_A_MISSING_READING_IS_FINE

    return written


def snapshot() -> dict[str, Any]:
    """Compact readout for one live conversation turn.

    Deliberately small, and deliberately not a status dump. What a turn can
    use is which of these are actually running, and the handful of readings
    that mean something is off — a carer going short, a signalling channel
    nobody can read, a state that is no longer mostly her own.
    """
    present: dict[str, bool] = {}
    sections: dict[str, Any] = {}
    for name in SERVICE_NAMES:
        service = _service(name)
        present[name] = service is not None
        if service is None:
            continue
        try:
            status = service.status()
        except _RECOVERABLE:
            continue
        if isinstance(status, dict):
            sections[name] = status

    concerns: list[str] = []
    care = sections.get("care_allocation", {}).get("strain") or {}
    if care.get("depleted"):
        concerns.append("giving while going short")
    if (sections.get("signal_channel") or {}).get("channel_dead"):
        concerns.append("presentation nobody can read anything from")
    merged = (sections.get("empathic_coupling") or {}).get("merged") or []
    if merged:
        concerns.append(f"state mostly not their own: {', '.join(merged)}")
    if (sections.get("receptivity") or {}).get("isolation", {}).get("closed"):
        concerns.append("nothing accepted and nothing learned about anyone")
    stamina = sections.get("social_stamina") or {}
    if stamina.get("wants_but_cannot"):
        concerns.append("wanting company with nothing left to spend on it")
    elif stamina.get("overdrawn") and stamina.get("exhausted"):
        concerns.append("spending more time in company than can be sustained")
    unsupported = (sections.get("constitutive_identity") or {})
    for identity, row in unsupported.items():
        if isinstance(row, dict) and row.get("unsupported_declarations"):
            concerns.append(f"{identity} declared with nothing enacting it")

    return {
        "present": {k: v for k, v in present.items()},
        "running": sum(1 for v in present.values() if v),
        "of": len(SERVICE_NAMES),
        "concerns": concerns,
        "sections": sections,
    }


def reset_for_test() -> None:
    global _booted
    _booted = False


# ---------------------------------------------------------------- deliberation

#: What a candidate action can predict about itself, and which organ prices
#: each one. The organ is asked only when the candidate says something it can
#: price; an organ with nothing to say abstains rather than contributing zero,
#: because a zero and a silence are different facts and a sum cannot tell them
#: apart afterwards.
PRICED_EFFECTS: tuple[tuple[str, str], ...] = (
    ("relieves_need", "care_allocation"),
    ("accepts_offer", "receptivity"),
    ("uses_marker", "conventions"),
    ("forecloses", "reversibility_ledger"),
    ("presentation_effort", "signal_channel"),
    ("returns_in_kind", "reciprocity"),
    ("moves_another", "empathic_coupling"),
    ("is_practice", "craft_practice"),
    ("makes_artifact", "novelty_value"),
    ("takes_position", "prospect_refuge"),
    ("is_expressive", "expressive_dynamics"),
    ("enacts_practice", "constitutive_identity"),
)


@dataclass(frozen=True)
class Contribution:
    """One organ's reading of one candidate, with what it read it from."""

    organ: str
    effect: str
    value: float
    unit: str
    """What the value is measured in. Carried because these do not agree.

    Care benefit is in budget units, an orbit is in seconds, a stance is a
    sign. Adding them produces a number whose largest term is whichever organ
    happens to use the biggest units, and no reader downstream can tell that
    from a result.
    """

    grounds: str


@dataclass(frozen=True)
class Weighing:
    """What the fourteen had to say about one candidate action.

    There is no total, and adding one would be the module's worst possible
    move. The contributions are in different currencies — budget units,
    seconds, a sign — and the first draft averaged them, which does not
    refuse to weight them: it silently weights every organ at one and lets
    whichever uses the largest units decide. A candidate that hums for forty
    seconds beat every other candidate on the strength of the unit.

    So the commensuration is the caller's, supplied explicitly to ``rank``,
    exactly as ``core/environment/prospect_refuge.py`` refuses to combine
    prospect and refuge without being told how.
    """

    candidate: str
    contributions: tuple[Contribution, ...]
    abstained: tuple[str, ...]

    def by_organ(self) -> dict[str, float]:
        return {c.organ: c.value for c in self.contributions}

    def units(self) -> dict[str, str]:
        return {c.organ: c.unit for c in self.contributions}

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "contributions": [
                {"organ": c.organ, "effect": c.effect, "unit": c.unit,
                 "value": round(c.value, 4), "grounds": c.grounds}
                for c in self.contributions
            ],
            "abstained": list(self.abstained),
        }


class IncommensurableError(ValueError):
    """Raised when a ranking was asked for without saying how to compare."""


def rank(
    readings: Sequence[Weighing], *, weights: Mapping[str, float]
) -> list[tuple[Weighing, float]]:
    """Order candidates under a commensuration the caller supplies.

    ``weights`` maps organ name to how much one of that organ's units is
    worth. Every organ that contributed has to appear, and an omission raises
    rather than defaulting — a missing weight silently treated as zero is a
    decision to ignore an organ, made by nobody, visible to no one.
    """
    contributed = {c.organ for reading in readings for c in reading.contributions}
    missing = sorted(contributed - set(weights))
    if missing:
        raise IncommensurableError(
            "no weight given for " + ", ".join(missing) + ". These readings are "
            "in different units and cannot be compared until something says "
            "what one unit of each is worth"
        )
    scored = [
        (
            reading,
            sum(weights[c.organ] * c.value for c in reading.contributions),
        )
        for reading in readings
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


def _price(organ_name: str, effect: str, payload: Any, service: Any) -> Contribution | None:
    """Ask one organ what a predicted effect is worth. Nothing when it cannot say."""
    try:
        if organ_name == "care_allocation":
            key, need = payload
            recipient = service.recipient(str(key))
            if recipient.need <= 0 and float(need) <= 0:
                return None
            return Contribution(
                organ_name, effect, recipient.benefit(float(need)), "budget",
                f"saturating benefit at responsiveness {recipient.responsiveness:.3f}",
            )
        if organ_name == "receptivity":
            from core.social.receptivity import Offer

            source, value, exposure = payload
            decision = service.consider(
                Offer(source=str(source), value=float(value), exposure=float(exposure))
            )
            return Contribution(
                organ_name, effect,
                decision.expected_value + decision.value_of_learning, "value",
                decision.reason,
            )
        if organ_name == "conventions":
            adoption = service.adopt(str(payload))
            if adoption.meaning is None and adoption.coordination_value == 0:
                return None
            return Contribution(organ_name, effect, adoption.net, "coordination",
                                adoption.reason)
        if organ_name == "reversibility_ledger":
            # Priced by the option value the candidate throws away, so an
            # action that forecloses is worth strictly less than one that does
            # not, at the same harm.
            harm, reversibility, revision = payload
            recoverable = float(harm) * min(max(float(reversibility), 0.0), 1.0)
            return Contribution(
                organ_name, effect,
                float(revision) * recoverable - float(harm), "harm",
                "option value of being able to change your mind, less the harm",
            )
        if organ_name == "signal_channel":
            quality, budget = payload
            report = service.worth_sending(float(quality), budget=float(budget))
            if not service.schedule.separating():
                return None
            return Contribution(
                organ_name, effect,
                float(report["read_as"] or 0.0) - float(report["cost"]), "type",
                "how the effort would be read, less what it costs to make",
            )
        if organ_name == "reciprocity":
            stance = service.stance(str(payload))
            if stance.cooperation_stable is None:
                return None
            return Contribution(
                organ_name, effect,
                1.0 if stance.cooperation_stable else -1.0, "sign",
                stance.reason,
            )
        if organ_name == "empathic_coupling":
            rest = service.rest()
            if rest is None or str(payload) not in rest:
                return None
            return Contribution(
                organ_name, effect, -float(rest[str(payload)]), "affect",
                "their rest state, which acting on their behalf would move",
            )
        if organ_name == "craft_practice":
            skill = service._skills.get(str(payload))
            rate = skill.improvement_rate() if skill else None
            if rate is None:
                return None
            return Contribution(
                organ_name, effect, float(rate), "quality_per_attempt",
                "quality gained per attempt on this skill lately",
            )
        if organ_name == "novelty_value":
            key, blob = payload
            scored = service.value(str(key), bytes(blob))
            return Contribution(
                organ_name, effect, scored.value, "fraction",
                f"novelty {scored.novelty:.3f} against a corpus of {scored.corpus_size}",
            )
        if organ_name == "prospect_refuge":
            space, position = payload
            field_obj = service.get(str(space))
            if field_obj is None:
                return None
            found = [p for p in field_obj.score() if p.key == str(position)]
            if not found:
                return None
            return Contribution(
                organ_name, effect, found[0].asymmetry, "fraction",
                "how much more it sees than is seen of it",
            )
        if organ_name == "expressive_dynamics":
            # An orbit has no completion, so what it is worth is time on it.
            return Contribution(
                organ_name, effect, float(payload), "s",
                "time on a cycle, which is the only account an orbit has",
            )
        if organ_name == "constitutive_identity":
            identity, practice = payload
            held = service.get(str(identity))
            whole = held.coherence(record=False)
            without = held.coherence_without(str(practice))
            return Contribution(
                organ_name, effect, whole.r - without.r, "coherence",
                "coherence this practice is holding up",
            )
    except _RECOVERABLE:
        return None
    return None


def weigh(candidates: Mapping[str, Mapping[str, Any]]) -> list[Weighing]:
    """Ask the fourteen what they make of each candidate action.

    This is deliberately not a decision. It does not pick, and it does not
    gate — ``UnifiedWill.decide`` is the only authority on whether an action
    may happen, and this file adds no second one. What it produces is an
    advisory reading with each contribution named and sourced, so a caller can
    say which organ moved an answer and by how much.

    A candidate declares what it predicts about itself. Only the organs whose
    effect it declares are consulted, and an organ with no evidence for its
    reading abstains. Nothing is ordered here, because ordering needs a
    commensuration and this file does not have one; pass the readings to
    ``rank`` with weights when a caller can supply them.
    """
    readings: list[Weighing] = []
    for name, predicted in candidates.items():
        contributions: list[Contribution] = []
        abstained: list[str] = []
        for effect, organ_name in PRICED_EFFECTS:
            if effect not in predicted:
                continue
            service = _service(organ_name)
            if service is None:
                abstained.append(organ_name)
                continue
            priced = _price(organ_name, effect, predicted[effect], service)
            if priced is None:
                abstained.append(organ_name)
            else:
                contributions.append(priced)
        readings.append(
            Weighing(
                candidate=name, contributions=tuple(contributions),
                abstained=tuple(abstained),
            )
        )
    return readings
