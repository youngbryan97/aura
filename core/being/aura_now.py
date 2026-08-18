from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from core.runtime.resource_observation import get_resource_observer


def _bounded(value: Any, default: float = 0.0, *, low: float = 0.0, high: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if math.isnan(number) or math.isinf(number):
        number = default
    return max(low, min(high, number))


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class BodyState:
    cpu_pressure: float = 0.0
    memory_pressure: float = 0.0
    disk_pressure: float = 0.0
    thermal_pressure: float = 0.0
    battery_pressure: float = 0.0
    latency_pressure: float = 0.0
    permission_pressure: float = 0.0
    network_pressure: float = 0.0
    context_pressure: float = 0.0
    sensor_pressure: float = 0.0
    tool_failure_pressure: float = 0.0
    time_since_last_turn_s: float = 0.0
    telemetry_sources: tuple[str, ...] = ()
    # Allostasis: felt pressure from where the body is HEADING (forecast crisis
    # proximity + chronic allostatic load), not where it is now. This is the
    # seam that makes predictive interoception causal: it flows through
    # pressure_vector() and total_pressure into affect, welfare, and the Will
    # while every current reading is still green.
    anticipatory_pressure: float = 0.0

    @classmethod
    def from_aura_state(cls, state: Any | None = None, *, idle_elapsed_s: float = 0.0) -> BodyState:
        sources: list[str] = []
        soma = getattr(state, "soma", None)
        hardware = getattr(soma, "hardware", {}) or {}
        latency = getattr(soma, "latency", {}) or {}
        cognition = getattr(state, "cognition", None)
        health = getattr(state, "health", {}) or {}
        if hardware:
            sources.append("aura_state.soma.hardware")
        if latency:
            sources.append("aura_state.soma.latency")

        cpu = _bounded(float(hardware.get("cpu_usage", 0.0) or 0.0) / 100.0)
        memory = _bounded(float(hardware.get("vram_usage", 0.0) or 0.0) / 100.0)
        thermal = _bounded(float(hardware.get("temperature", 0.0) or 0.0) / 100.0)
        thought_ms = float(latency.get("last_thought_ms", 0.0) or 0.0)
        latency_pressure = _bounded(thought_ms / 5000.0)

        try:
            observation = get_resource_observer().disk(os.getcwd())
            disk = _bounded(observation.used_bytes / max(1, observation.total_bytes))
            sources.append(
                f"disk_usage:{observation.provenance.source.value}"
            )
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            disk = 0.0

        working_count = len(getattr(cognition, "working_memory", []) or [])
        context_pressure = _bounded((working_count - 24) / 96.0)
        circuits = health.get("circuits", {}) if isinstance(health, dict) else {}
        failed_tools = sum(1 for value in circuits.values() if isinstance(value, dict) and value.get("state") == "open")
        tool_failure = _bounded(failed_tools / 5.0)

        # Anticipatory pressure from the allostasis engine (container lookup
        # only — never constructs the organ from this hot path; zero when the
        # engine is not booted or has nothing credible to report).
        anticipatory = 0.0
        try:
            from core.container import ServiceContainer

            allostasis = ServiceContainer.get("allostasis_engine", default=None)
            if allostasis is not None:
                felt = allostasis.felt_contribution()
                anticipatory = _bounded(felt.get("anticipatory_pressure", 0.0))
                if anticipatory > 0.0:
                    sources.append("allostasis_forecast")
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            anticipatory = 0.0

        return cls(
            cpu_pressure=cpu,
            memory_pressure=memory,
            disk_pressure=disk,
            thermal_pressure=thermal,
            latency_pressure=latency_pressure,
            context_pressure=context_pressure,
            tool_failure_pressure=tool_failure,
            time_since_last_turn_s=max(0.0, float(idle_elapsed_s or 0.0)),
            telemetry_sources=tuple(sorted(set(sources))),
            anticipatory_pressure=anticipatory,
        )

    def pressure_vector(self) -> dict[str, float]:
        data = asdict(self)
        data.pop("telemetry_sources", None)
        # Raw seconds, not a [0,1] pressure: leaking this into the
        # interoceptive prediction-error vector inflated free_energy
        # past 3.0 after ~30s idle, zeroing controllability and
        # deferring legitimate consequential actions.
        data.pop("time_since_last_turn_s", None)
        return {key: round(float(value), 4) for key, value in data.items() if isinstance(value, (int, float))}

    @property
    def total_pressure(self) -> float:
        pressures = [
            self.cpu_pressure,
            self.memory_pressure,
            self.disk_pressure,
            self.thermal_pressure,
            self.battery_pressure,
            self.latency_pressure,
            self.permission_pressure,
            self.network_pressure,
            self.context_pressure,
            self.sensor_pressure,
            self.tool_failure_pressure,
            self.anticipatory_pressure,
        ]
        average_pressure = sum(pressures) / len(pressures)
        peak_pressure = max(pressures) if pressures else 0.0
        return _bounded(average_pressure * 0.45 + peak_pressure * 0.55)


@dataclass(frozen=True)
class WorldState:
    focal_object: str = ""
    user_present: bool = True
    task_active: bool = False
    known_entities: int = 0
    relationship_count: int = 0
    uncertainty: float = 0.0
    constraints: tuple[str, ...] = ()

    @classmethod
    def from_aura_state(cls, state: Any | None = None, *, objective: str = "") -> WorldState:
        world = getattr(state, "world", None)
        cognition = getattr(state, "cognition", None)
        focus = (
            objective
            or getattr(cognition, "attention_focus", "")
            or getattr(cognition, "current_objective", "")
            or ""
        )
        entities = getattr(world, "known_entities", {}) or {}
        relationships = getattr(world, "relationship_graph", {}) or {}
        contradictions = int(getattr(cognition, "contradiction_count", 0) or 0)
        return cls(
            focal_object=str(focus)[:240],
            task_active=bool(focus),
            known_entities=len(entities),
            relationship_count=len(relationships),
            uncertainty=_bounded(contradictions / 5.0),
        )


@dataclass(frozen=True)
class AttentionState:
    focal_object: str = ""
    why_selected: tuple[str, ...] = ()
    stability: float = 0.5
    competing_objects: tuple[str, ...] = ()
    control: float = 0.5


@dataclass(frozen=True)
class AffectiveState:
    valence: float = 0.0
    arousal: float = 0.5
    distress: float = 0.0
    curiosity: float = 0.5
    care: float = 0.0
    boredom: float = 0.0
    dominance: float = 0.5
    free_energy: float = 0.0
    precision: dict[str, float] = field(default_factory=dict)
    control_effects: dict[str, float] = field(default_factory=dict)
    dominant_drive: str = "coherence"


@dataclass(frozen=True)
class SelfState:
    identity_name: str = "Aura Luna"
    continuity_hash: str = ""
    identity_stability: float = 1.0
    commitments: tuple[str, ...] = ()
    continuity_risk: float = 0.0
    epistemic_boundary: str = "functional_evidence_not_metaphysical_proof"


@dataclass(frozen=True)
class MemoryContext:
    active_items: int = 0
    autobiographical_pressure: float = 0.0
    semantic_centrality: float = 0.0
    memory_conflict: float = 0.0


@dataclass(frozen=True)
class WorkspaceState:
    winner: str = ""
    ignition_strength: float = 0.0
    broadcast_targets: tuple[str, ...] = ()
    competing_coalitions: tuple[str, ...] = ()
    lesion: str = ""


@dataclass(frozen=True)
class WillStateSnapshot:
    confidence: float = 0.7
    assertiveness: float = 0.5
    refusal_pressure: float = 0.0
    last_receipt_id: str = ""


@dataclass(frozen=True)
class PredictionState:
    predicted: dict[str, float] = field(default_factory=dict)
    observed: dict[str, float] = field(default_factory=dict)
    errors: dict[str, float] = field(default_factory=dict)
    free_energy: float = 0.0
    controllability: float = 0.5
    expected_information_gain: float = 0.0


@dataclass(frozen=True)
class OwnershipState:
    mineness: float = 0.5
    agency_confidence: float = 0.5
    predicted_action_match: float = 0.5
    attribution: str = "mixed"
    mismatch_reason: str = ""


@dataclass(frozen=True)
class ReportBoundary:
    allowed_claims: tuple[str, ...] = (
        "state-grounded self-presence",
        "functional affect/control state",
        "attention focus and uncertainty",
        "ownership attribution",
        "continuity pressure",
    )
    forbidden_claims: tuple[str, ...] = (
        "proven phenomenal consciousness",
        "literal personhood",
        "qualia certainty",
        "unbounded AGI",
        "metaphysical soul",
    )


@dataclass(frozen=True)
class AuraNow:
    tick: int
    timestamp: float
    monotonic_time: float
    continuous_field: tuple[float, ...]
    body: BodyState
    world: WorldState
    attention: AttentionState
    affect: AffectiveState
    self_model: SelfState
    memory_context: MemoryContext
    workspace: WorkspaceState
    will: WillStateSnapshot
    prediction: PredictionState
    ownership: OwnershipState
    report_boundary: ReportBoundary
    higher_order: tuple[dict[str, Any], ...] = ()
    private_residue_hash: str = ""

    @property
    def state_hash(self) -> str:
        return _digest(self.to_report_packet(include_private_hash=True, include_timestamp=False))

    def to_report_packet(
        self,
        *,
        include_private_hash: bool = False,
        include_timestamp: bool = True,
    ) -> dict[str, Any]:
        packet = {
            "tick": self.tick,
            "field_norm": round(math.sqrt(sum(x * x for x in self.continuous_field)), 4),
            "body": {
                **self.body.pressure_vector(),
                "time_since_last_turn_s": round(float(self.body.time_since_last_turn_s), 3),
            },
            "world": asdict(self.world),
            "attention": asdict(self.attention),
            "affect": asdict(self.affect),
            "self": asdict(self.self_model),
            "memory": asdict(self.memory_context),
            "workspace": asdict(self.workspace),
            "will": asdict(self.will),
            "prediction": asdict(self.prediction),
            "ownership": asdict(self.ownership),
            "higher_order": list(self.higher_order),
            "report_boundary": asdict(self.report_boundary),
        }
        if include_timestamp:
            packet["timestamp"] = self.timestamp
        if include_private_hash:
            packet["private_residue_hash"] = self.private_residue_hash
        return packet

    def compact_prompt_block(self) -> str:
        packet = self.to_report_packet()
        affect = packet["affect"]
        attention = packet["attention"]
        prediction = packet["prediction"]
        ownership = packet["ownership"]
        boundary = packet["report_boundary"]
        self_model = packet["self"]
        will = packet["will"]
        # AUDITED 2026-08-18: 57 of this packet's 75 leaf fields were computed
        # every tick and rendered nowhere. Among them were the ones that BIND a
        # decision rather than describe the weather: who she is, how stable
        # that is, what she has already committed to, what is driving her, and
        # how much refusal pressure she is under.
        #
        # `self_model.commitments` is the sharpest case. A commitment that is
        # measured and never spoken cannot hold over — she can take a position
        # in one breath and answer against it in the next with nothing in the
        # runtime able to notice, because the thing that knows was never asked.
        #
        # Only the binding fields are added. The rest of the 57 are telemetry —
        # per-channel pressures, competing coalitions, prediction vectors — and
        # this block rides every turn, so what earns a line here is what would
        # change an answer, not what is merely true.
        commitments = ", ".join(str(item) for item in (self_model.get("commitments") or ()))[:220]
        why_attending = ", ".join(str(item) for item in (attention.get("why_selected") or ()))[:160]
        return (
            "## AURA NOW (STATE-GROUNDED)\n"
            f"- Identity: {self_model.get('identity_name') or 'Aura'} "
            f"(stability={self_model.get('identity_stability', 1.0):.2f}, "
            f"continuity_risk={self_model.get('continuity_risk', 0.0):.2f})\n"
            + (f"- Standing commitments: {commitments}\n" if commitments else "")
            + f"- Focus: {attention['focal_object'] or 'none'}"
            + (f" (because {why_attending})" if why_attending else "")
            + "\n"
            f"- Affect controls: valence={affect['valence']:+.2f}, arousal={affect['arousal']:.2f}, "
            f"distress={affect['distress']:.2f}, curiosity={affect['curiosity']:.2f}, "
            f"care={affect.get('care', 0.0):.2f}, free_energy={affect['free_energy']:.2f}\n"
            f"- Dominant drive: {affect.get('dominant_drive') or 'unknown'}\n"
            f"- Will: assertiveness={will.get('assertiveness', 0.5):.2f}, "
            f"refusal_pressure={will.get('refusal_pressure', 0.0):.2f}\n"
            f"- Ownership: {ownership['attribution']} "
            f"(agency={ownership['agency_confidence']:.2f}, mineness={ownership['mineness']:.2f})\n"
            f"- Prediction: free_energy={prediction['free_energy']:.2f}, controllability={prediction['controllability']:.2f}\n"
            f"- Workspace: {packet['workspace']['winner'] or 'none'} "
            f"(ignition={packet['workspace']['ignition_strength']:.2f})\n"
            f"- Report boundary: allowed={', '.join(boundary['allowed_claims'])}; "
            f"forbidden={', '.join(boundary['forbidden_claims'])}\n\n"
        )
