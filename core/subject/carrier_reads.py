"""What K's transition reads beyond its ten domains, which the carrier holds as its own.

The closure test asks whether K's future needs anything outside K. On seed 23
it did: thirteen units outside K's ten domains changed K within two turns when
their state alone was swapped in from another anchor, through hybrid snapshots
and nothing else (25 October). A core whose transition reads state it does not
contain is drawn too small, so the carrier is K plus that state: K*.

Membership is decided by mechanism, never by which columns leaked. An attribute
is held when swapping it alone changes K, or changes an attribute already held,
within two turns at some anchor, iterated until a round adds nothing
(`organ_fixed_point_v3.py`, `attribute_interchange.py`). Each held attribute
belongs to the domain whose function it serves, so a cut that holds a domain
holds it too.

Left out by rule, not by result:

- Wall-clock instants (an attribute that is a moment, like a last-update time).
  Swapping one injects an elapsed time no run produces, and a restore already
  treats instants as anchors of the clock rather than as state.
- Tables of the organ's own methods, which are wiring (`core/subject/snapshot.py`).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["HELD", "Held", "held_attributes_of", "held_by_domain", "read_held"]


@dataclass(frozen=True, slots=True)
class Held:
    unit: str
    attribute: str
    domain: str


#: K*'s additions to K. From the attribute-level fixed point on the seed-23
#: anchors at the commit named in the ISC-v6 addendum of 25 October.
HELD: tuple[Held, ...] = (
    # Affect and conation: her appraisal, her affect engine's markers, the
    # buffer her affect grounding reads, and her homeostatic drives.
    Held("AffectUpdatePhase", "appraisal", "A"),
    Held("affect_engine", "markers", "A"),
    Held("affect_grounding", "_buf", "A"),
    Held("homeostasis", "_setpoints", "A"),
    Held("homeostasis", "integrity", "A"),
    Held("homeostasis", "persistence", "A"),
    Held("homeostasis", "sovereignty", "A"),
    # Swapped alone it moves K through the self-predictive loop's weights
    # (the interventional test on seed 23's anchors), and it was the carrier's
    # second-largest observational leak.
    Held("homeostasis", "metabolism", "A"),
    # Body: interoceptive channels and neurochemistry.
    Held("embodied_interoception", "channels", "I"),
    Held("neurochemical_system", "chemicals", "I"),
    Held("neurochemical_system", "_control", "I"),
    Held("neurochemical_system", "_tick_count", "I"),
    # Her running sense of load, which sets the threat her heartbeat signals
    # (the interventional test on seed 23's anchors, at every anchor).
    Held("existential_stakes", "_rolling_cpu_load", "I"),
    # Self-state: the beliefs she holds, her beliefs about herself among them
    # (the interventional test, at every anchor once the order of swaps let it).
    Held("belief_authority", "_beliefs", "S"),
    # Recurrent cognition: the self-predictive loop, binding oscillations, STDP traces.
    Held("closed_causal_loop", "_loop_state", "C"),
    Held("closed_causal_loop", "_predictor", "C"),
    Held("oscillatory_binding", "_gamma_at_theta_peak", "C"),
    Held("oscillatory_binding", "_gamma_at_theta_trough", "C"),
    Held("oscillatory_binding", "_gamma_phase", "C"),
    Held("oscillatory_binding", "_mesh_association_energy", "C"),
    Held("oscillatory_binding", "_mesh_executive_energy", "C"),
    Held("oscillatory_binding", "_mesh_sensory_energy", "C"),
    Held("oscillatory_binding", "_phase_reports", "C"),
    # Moves the held gamma readings at every anchor (the interventional test).
    Held("oscillatory_binding", "_theta_phase", "C"),
    Held("stdp_engine", "_eligibility", "C"),
    Held("stdp_engine", "_mesu_var", "C"),
    # Workspace and unity: coherence, the unity binding, the moments she binds.
    Held("coherence_report", "overall_coherence", "G"),
    Held("unity_runtime", "_last_unity_state", "G"),
    Held("unity_runtime", "temporal_binding", "G"),
    Held("phenomenal_now_engine", "_recent", "G"),
    Held("consciousness", "heartbeat", "G"),
    Held("qualia_synthesizer", "q_norm", "G"),
)


def held_attributes_of(unit: str) -> frozenset[str]:
    """The attributes of one unit that K* holds."""
    return frozenset(entry.attribute for entry in HELD if entry.unit == unit)


def held_by_domain(domain: str) -> tuple[Held, ...]:
    return tuple(entry for entry in HELD if entry.domain == domain)


def _units(kernel: Any) -> dict[str, Any]:
    """Every unit K* holds part of, by name: built services and the kernel's phases."""
    from core.subject.snapshot import _built_services

    out: dict[str, Any] = {}
    services = _built_services()
    phases = {type(phase).__name__: phase for phase in getattr(kernel, "_phases", []) or []}
    for unit in {entry.unit for entry in HELD}:
        found = services.get(unit)
        out[unit] = found if found is not None else phases.get(unit)
    return out


def read_held(kernel: Any, units: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Every held attribute's numbers, named `held.<unit>.<attribute>...`.

    Read by the periphery's own reader so the two sides of the closure test
    see the same numbers in the same way, and descended into from the
    attribute itself, so a field below the walk's depth limit is read here
    rather than silently left out of both.
    """
    from core.subject.closure import _contents, _is_array, _is_clock, _numbers, _sketch_array

    import math

    from core.subject.snapshot import _built_services

    out: dict[str, float] = {}
    live = dict(units) if units is not None else _units(kernel)
    # Each attribute's own numbers, never another unit's reached through a
    # reference it holds. Her heartbeat keeps the qualia synthesizer, homeostasis
    # and temporal binding as fields; read whole, it put their state into K*, and
    # swapping any of them then "moved K*" without moving K (seed 23, 25 October).
    # The fork skips such references for the same reason (`_carried_by_name`).
    others = {id(service) for service in _built_services().values() if service is not None}
    others |= {id(phase) for phase in getattr(kernel, "_phases", []) or []}
    for entry in HELD:
        organ = live.get(entry.unit)
        if organ is None:
            continue
        value = getattr(organ, entry.attribute, None)
        name = f"held.{entry.unit}.{entry.attribute}"
        if isinstance(value, bool):
            out[name] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            number = float(value)
            if math.isfinite(number) and not _is_clock(number):
                out[name] = number
        elif isinstance(value, (list, tuple, dict, set)) or type(value).__name__ == "deque":
            out[f"{name}#"] = float(len(value))
            _contents(value, name, out, seen=set(others))
        elif _is_array(value):
            summary = _sketch_array(value)
            if summary is not None:
                for part, number in summary.items():
                    out[f"{name}.{part}"] = number
        elif hasattr(value, "__dict__") and id(value) not in others:
            _numbers(value, name, out, depth=0, ceiling=None, seen=set(others))
    return out
