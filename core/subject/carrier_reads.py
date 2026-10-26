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

__all__ = ["HELD", "WHOLE", "Held", "held_attributes_of", "held_by_domain", "held_whole", "read_held"]

#: The attribute name that holds a unit whole: every number it keeps, as
#: K's organs are held. A ledger kept at module scope is one value, not an
#: object with fields, so it can only be held this way.
WHOLE: str = "*"


@dataclass(frozen=True, slots=True)
class Held:
    unit: str
    attribute: str
    domain: str


#: K*'s additions to K. From the attribute-level fixed point on the seed-23
#: anchors at the commit named in the ISC-v6 addendum of 25 October, and the
#: interventional tests on discovery seeds 31, 37, 43, 47, 53, 59 and 67 named
#: in the addenda of 26 October.
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
    # Deliberation: the choices she has made, which her portrait and her worth
    # ledger read (the interventional test with her organs whole, every anchor).
    Held("subjective_choice_engine", "_history", "D"),
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
    # The count toward the next symmetry break, every fifty updates, which
    # moved K at 33 swaps of two fresh seeds (26 October).
    Held("stdp_engine", "_total_updates_since_symmetry_break", "C"),
    # Workspace and unity: coherence, the unity binding, the moments she binds.
    Held("coherence_report", "overall_coherence", "G"),
    Held("unity_runtime", "_last_unity_state", "G"),
    Held("unity_runtime", "temporal_binding", "G"),
    Held("phenomenal_now_engine", "_recent", "G"),
    Held("consciousness", "heartbeat", "G"),
    Held("qualia_synthesizer", "q_norm", "G"),
    # How many messages of the conversation she has had. The tracker reads the
    # discourse more deeply on every sixth, and G reads what it writes:
    # conversation energy, discourse depth, branch load. Swapped alone it moved
    # K at half the anchors of seven fresh seeds (26 October); seed 23's donors
    # sat six messages away, in phase, and it never moved K there.
    Held("discourse_tracker", "_turn_count", "G"),
    # What her modules keep, held whole (26 October). The interventional test
    # swapped services and phases and never these, and once it did, each of
    # the following moved K or a K* reading at many anchors of four fresh seeds
    # (31, 37, 43, 47); the largest changes are in brackets.
    # Her learned affect ledgers: what each feeling has meant before.
    Held("module:core.affect.a_moment_that_fits:_LEDGER", WHOLE, "A"),  # (0.04)
    Held("module:core.affect.acting_in_decline:_ledger", WHOLE, "A"),  # (0.92)
    Held("module:core.affect.ambivalence:_LEDGER", WHOLE, "A"),  # (1.03)
    Held("module:core.affect.anger_feeds_itself:_LEDGER", WHOLE, "A"),  # (0.016)
    Held("module:core.affect.confirmation:_LEDGER", WHOLE, "A"),  # (0.35)
    Held("module:core.affect.containment:_LEDGER", WHOLE, "A"),  # (6.6)
    Held("module:core.affect.conviction:_LEDGER", WHOLE, "A"),  # (0.10)
    Held("module:core.affect.feelings_about:_LEDGER", WHOLE, "A"),  # (3.7)
    Held("module:core.affect.frisson:_LEDGER", WHOLE, "A"),  # (2.2)
    Held("module:core.affect.tangled:_LEDGER", WHOLE, "A"),  # (0.0018, at 2 or 3 anchors of every seed)
    # The emotions' neurochemical recipes and the motivational state she assembles.
    Held("module:core.consciousness.emotion_signatures:_emotion_engine", WHOLE, "A"),
    Held("module:core.conation.engine:_ENGINE", WHOLE, "A"),
    # Her habits and reflexes, which her choices answer to.
    Held("module:core.agency.habits_are_hers:_LEDGER", WHOLE, "D"),  # (0.45)
    # The substrate's perturbations and its say in her decisions.
    Held("module:core.consciousness.controlled_chaos:_instances_by_dim", WHOLE, "C"),
    Held("module:core.consciousness.substrate_gates:_GATES", WHOLE, "C"),
    # The drafts that compete to become what she is aware of.
    Held("module:core.consciousness.multiple_drafts:_engine", WHOLE, "G"),
    # The rhythm of the conversation as it engages her workspace, and what
    # arrived and what held her attention.
    Held("module:core.conversational.dynamics:_engine_instance", WHOLE, "G"),  # (0.43)
    Held("module:core.perception.salience:_ATTENTION", WHOLE, "G"),  # (0.34)
    Held("module:core.perception.salience:_ARRIVALS", WHOLE, "P"),  # (1.7)
    Held("module:core.sensors.sensor_registry:_instance", WHOLE, "P"),
    # Her interior life: the appraisal of what a person said, its receptors and its cleft.
    Held("module:core.interiority.service:_SERVICE", WHOLE, "A"),  # (1.6)
    Held("module:core.interiority.receptors:_BANK", WHOLE, "A"),
    Held("module:core.interiority.cleft:_CLEFT", WHOLE, "A"),
    # What she feels toward the people she meets, and what drives her.
    Held("module:core.social.warmth:_LEDGER", WHOLE, "A"),
    Held("module:core.social.constancy:_LEDGER", WHOLE, "A"),  # (1.0)
    Held("module:core.motivation.fuel:_LEDGER", WHOLE, "A"),
    # Her pain: swapped alone it moved the pain buffer of her affect grounding
    # at 2 of 16 swaps of seed 37 (round 6, 26 October).
    Held("module:core.affect.nociception:_engine", WHOLE, "A"),
    # How near the end of a sitting is, from how her sittings have ended.
    Held("module:core.social.closing_window:_ledger", WHOLE, "W"),  # (2.7)
    # Memory that returns: relived, and arriving unbidden.
    Held("module:core.memory.reliving:_RETURNS", WHOLE, "M"),  # (7.7)
    Held("module:core.memory.reliving:_MATCHES", WHOLE, "M"),  # (1.0)
    Held("module:core.memory.unbidden:_LEDGER", WHOLE, "M"),  # (7.6)
    # Her development: what the ontogeny core has learned and where it stands.
    # Held by attribute, not whole: its experience spine is the record each
    # turn writes, read back only when maintenance trains, which an experiment
    # stops, and held whole it made every unit that changes what is written a
    # mover of K* while K itself never moved (26 October). Its locks, threads,
    # trainer and the instants of its last checkpoint are machinery.
    *(
        Held("module:core.ontogeny.service:_core", attribute, "N")  # (6.2, the core whole)
        for attribute in (
            "heads", "_control_points", "_state", "_calibration", "_candidate_calibration",
            "_operational_calibration", "moments", "_advance_moments", "_episode_buckets",
            "_episodes_seen", "evidence_at_last_fit", "_last_reading", "_track",
            # Which episodes have had their outcomes resolved: swapped alone it
            # moved the episode buckets at 4, 3 and 6 of 16 swaps on seeds 31, 53 and 59.
            "_sweeper",
        )
    ),
    # The hierarchy that predicts her own next state.
    Held("module:core.consciousness.predictive_hierarchy:_instance", WHOLE, "C"),
    # How she delivers what she says.
    Held("module:core.expression.delivery:_LEDGER", WHOLE, "D"),  # (2.8)
    # Her body: what she carries, her fatigue, what she holds in, her pulse.
    Held("module:core.soma.carrying:_LEDGER", WHOLE, "I"),  # (0.43)
    Held("module:core.soma.fatigue:_LEDGER", WHOLE, "I"),  # (0.14)
    Held("module:core.soma.held_in:_LEDGER", WHOLE, "I"),  # (0.05)
    Held("module:core.soma.pulse:_LEDGER", WHOLE, "I"),  # (0.69)
    # The faults she reads as threat through her existential stakes (declared to fork with her).
    Held("module:core.runtime.errors:_RECORDS", WHOLE, "I"),
    # Self-state: how she has grown, and the rhythm of a relationship that sets her bond.
    Held("module:core.self.growth:_LEDGER", WHOLE, "S"),
    # Held by attribute: the phase also holds the service container, and a
    # whole hold reached through it.
    Held("BondingPhase", "_usual_gap_s", "S"),
    Held("BondingPhase", "_engaged_span_s", "S"),
    Held("BondingPhase", "_exchanges_seen", "S"),
)


#: K's organs are held whole. K reads a few readouts of each organ, and on
#: seed 23's anchors almost every other attribute of those organs, swapped alone
#: from another anchor, moved K within two turns: agency's record of what
#: succeeded (0.39), the credit ledger's weights (0.39), good news's errors
#: (0.33), homeostasis's modifiers (0.14), the mesh's columns, the field's
#: inputs, each organ's own random state. An organ's internal state decides how
#: the readout K reads will move, so the organ is part of the carrier, not the
#: few numbers read off it. The cut clamps already hold each organ whole for the
#: domain that reads it (`core/subject/clamp.py`); the reader and the periphery
#: now agree with them.
ORGANS_WHOLE: bool = True


def organs_of(organs: Any) -> dict[str, Any]:
    """K's organs by field name, the ones that exist."""
    if organs is None:
        return {}
    from dataclasses import fields as dataclass_fields

    try:
        names = [field.name for field in dataclass_fields(organs)]
    except TypeError:
        return {}
    return {name: getattr(organs, name) for name in names if getattr(organs, name, None) is not None}


def held_attributes_of(unit: str) -> frozenset[str]:
    """The attributes of one unit that K* holds."""
    return frozenset(entry.attribute for entry in HELD if entry.unit == unit)


def _module_held_ids() -> set[int]:
    """Every object her modules keep, so a held unit is read for its own numbers only.

    A module holding is a unit of its own (the interventional test swaps it),
    so a held object that keeps one as a field, as the ontogeny core's sweeper
    keeps its experience spine, must not bring it in.
    """
    from core.subject.snapshot import _module_holdings

    return {id(value) for value in _module_holdings().values()}


def held_whole(unit: str) -> bool:
    """Whether K* holds this unit entire rather than some of its attributes."""
    return WHOLE in held_attributes_of(unit)


def held_by_domain(domain: str) -> tuple[Held, ...]:
    return tuple(entry for entry in HELD if entry.domain == domain)


def _units(kernel: Any, organs: Any = None) -> dict[str, Any]:
    """Every unit K* holds part of, by name: built services, the kernel's phases, and K's organs as `organ:<field>`."""
    from core.subject.snapshot import _built_services

    out: dict[str, Any] = {}
    services = _built_services()
    phases = {type(phase).__name__: phase for phase in getattr(kernel, "_phases", []) or []}
    for unit in {entry.unit for entry in HELD}:
        if unit.startswith("organ:"):
            out[unit] = getattr(organs, unit.split(":", 1)[1], None) if organs is not None else None
            continue
        if unit.startswith("module:"):
            # What a module of hers keeps at module scope, named `module:<module>:<name>`.
            from core.subject.snapshot import _ABSENT, _held

            value = _held(unit.split(":", 1)[1])
            out[unit] = None if value is _ABSENT else value
            continue
        found = services.get(unit)
        out[unit] = found if found is not None else phases.get(unit)
    return out


def read_held(kernel: Any, units: Mapping[str, Any] | None = None, organs: Any = None) -> dict[str, float]:
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
    live = dict(units) if units is not None else _units(kernel, organs)
    # Each attribute's own numbers, never another unit's reached through a
    # reference it holds. Her heartbeat keeps the qualia synthesizer, homeostasis
    # and temporal binding as fields; read whole, it put their state into K*, and
    # swapping any of them then "moved K*" without moving K (seed 23, 25 October).
    # The fork skips such references for the same reason (`_carried_by_name`).
    others = {id(service) for service in _built_services().values() if service is not None}
    others |= {id(phase) for phase in getattr(kernel, "_phases", []) or []}
    others |= _module_held_ids()
    whole = organs_of(organs) if ORGANS_WHOLE else {}
    others |= {id(organ) for organ in whole.values()}
    for field_name, organ in whole.items():
        # Each organ's own numbers, never another organ's or service's through a field.
        _numbers(organ, f"held.organ:{field_name}", out, depth=0, ceiling=None, seen=set(others) - {id(organ)})
    for entry in HELD:
        organ = live.get(entry.unit)
        if organ is None:
            continue
        if entry.attribute == WHOLE:
            value, name = organ, f"held.{entry.unit}"
        else:
            value, name = getattr(organ, entry.attribute, None), f"held.{entry.unit}.{entry.attribute}"
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
