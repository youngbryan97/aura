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
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Phenomena.Wiring")

#: Container keys for the fourteen. The order is the order they are reported
#: in, which is roughly the order they were built rather than any hierarchy.
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
