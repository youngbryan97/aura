"""The ten domains, and what a reading of each one is.

Everything downstream measures a matrix. This module says what a column of
that matrix is, and it is the place where the whole battery can be quietly
falsified, so it holds to three rules.

A **state variable** is a live number whose value changes what the system
computes next. Not a source file, not a module name, not generated prose. The
test is causal: if writing a different value into it changes the future, it is
state; if nothing reads it, it is a log line. Every feature below names the
attribute it reads, and `perturb` writes to that same attribute, so a feature
that no phase consults will show up as a domain nothing can move rather than
as a number that looks fine.

The **core** is a subset of the machine, not the machine. Allocator arenas,
HTTP buffers and log ring positions are all state by the definition above and
none of them belong to K. What belongs is the smallest cognitive state that
explains the continuing agent:

    P(K_{t+1} | X_t, E_t) ~ P(K_{t+1} | K_t, E_t)

That approximation is an empirical claim, not a definition, and
`core.subject.closure` measures it rather than assuming it.

The **schema is fixed**. Each domain has a declared, ordered list of named
features and a width that does not depend on the state being read. A vector
whose length changes with content cannot be compared across time, and a
distance computed between two such vectors is a measurement of the schema.

Half of each domain is read from `AuraState` and half from the live organs —
the global workspace, the liquid substrate, the free-energy engine, the self
model, the unified world model, the ontogenetic reservoir. The first version of
this file read only `AuraState` and reported that the domains barely influence
each other, which was true of the summary fields and false of the organism:
most of the coupling happens between organs and only its residue reaches the
state object. An organ that is absent reads as zeros, which is honest and is
visible as a flat column rather than as a plausible number.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "DOMAINS",
    "DOMAIN_NAMES",
    "FAST_DOMAINS",
    "SLOW_DOMAINS",
    "CoreState",
    "Organs",
    "Schema",
    "domain_width",
    "feature_names",
    "perturb",
    "perturb_organs",
    "perturbable",
    "read_core_state",
    "schema",
]

#: The ten domains, in the order their slices appear in K_t.
DOMAINS: tuple[str, ...] = ("P", "I", "A", "G", "C", "S", "M", "W", "D", "N")

DOMAIN_NAMES: dict[str, str] = {
    "P": "perception",
    "I": "interoception/body",
    "A": "affect/conation",
    "G": "workspace/attention",
    "C": "recurrent cognition",
    "S": "self-state",
    "M": "active memory",
    "W": "world model",
    "D": "deliberation/intention",
    "N": "ontogenetic/developmental state",
}

#: Which clocks a domain runs on. The cross-timescale test needs the split
#: declared before the measurement, not chosen after seeing which way it came
#: out. N is the lifetime reservoir; S and M carry across turns; the rest move
#: within a turn.
FAST_DOMAINS: tuple[str, ...] = ("P", "I", "A", "G", "C", "D")
SLOW_DOMAINS: tuple[str, ...] = ("S", "M", "W", "N")


# ── the schema ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Schema:
    """One domain's features, in order, each with the attribute it reads."""

    domain: str
    features: tuple[str, ...]
    sources: tuple[str, ...]

    @property
    def width(self) -> int:
        return len(self.features)


def _sch(domain: str, pairs: Sequence[tuple[str, str]]) -> Schema:
    return Schema(
        domain=domain,
        features=tuple(name for name, _ in pairs),
        sources=tuple(source for _, source in pairs),
    )


@dataclass(frozen=True, slots=True)
class Organs:
    """The live objects a reading needs besides `AuraState`.

    Every field may be None. None means the organ was not running, which is a
    different thing from an organ that was running and had nothing to say, and
    the difference shows up as a column that never moves.
    """

    workspace: Any = None
    substrate: Any = None
    free_energy: Any = None
    self_model: Any = None
    world_model: Any = None
    ontogeny: Any = None
    agency: Any = None
    self_prediction: Any = None

    @classmethod
    def live(cls) -> Organs:
        """Whatever is registered right now. Missing organs stay None."""

        def service(name: str) -> Any:
            try:
                from core.container import ServiceContainer

                return ServiceContainer.get(name, default=None)
            except Exception:  # noqa: BLE001 - an absent container is an absent organ
                return None

        def runtime(name: str) -> Any:
            try:
                from core.runtime.service_registry import get_runtime_service

                return get_runtime_service(name, default=None)
            except Exception:  # noqa: BLE001
                return None

        agency = None
        try:
            from core.agency.authorship import get_agency_ledger

            agency = get_agency_ledger()
        except Exception:  # noqa: BLE001
            agency = None
        return cls(
            workspace=runtime("global_workspace"),
            substrate=runtime("conscious_substrate"),
            free_energy=service("free_energy_engine"),
            self_model=service("self_model"),
            world_model=service("unified_world_model"),
            agency=agency,
            self_prediction=runtime("self_prediction"),
        )


def _call(obj: Any, name: str, default: Any = None) -> Any:
    """Call a reader on an organ, and treat any failure as no reading."""
    if obj is None:
        return default
    method = getattr(obj, name, None)
    if method is None:
        return default
    try:
        return method()
    except Exception:  # noqa: BLE001 - an organ that raises has told us nothing
        return default


#: The emotions read into A, in a fixed order. Chosen from the Plutchik
#: primaries that `AffectVector.emotions` always carries, so the width never
#: depends on which emotions happen to be present.
_EMOTIONS: tuple[str, ...] = (
    "joy",
    "trust",
    "fear",
    "surprise",
    "sadness",
    "disgust",
    "anger",
    "anticipation",
    "curiosity",
    "frustration",
)

#: The motivation budgets read into D, likewise fixed.
_DRIVES: tuple[str, ...] = ("social", "curiosity", "rest", "creation")

#: Cognitive modes, one-hot into C.
_MODES: tuple[str, ...] = ("reactive", "deliberate", "dreaming", "dormant")

_SCHEMAS: dict[str, Schema] = {
    "P": _sch(
        "P",
        (
            ("percept_load", "world.recent_percepts"),
            ("percept_recency", "world.recent_percepts[-1].timestamp"),
            ("percept_sources", "world.recent_percepts[*].source"),
            ("percept_novelty", "world.recent_percepts[*].content"),
            ("spatial_present", "world.spatial_context"),
            ("entity_load", "world.known_entities"),
            ("objective_len", "cognition.current_objective"),
            ("objective_hash", "cognition.current_objective"),
        ),
    ),
    "I": _sch(
        "I",
        (
            ("cpu", "soma.hardware.cpu_usage"),
            ("vram", "soma.hardware.vram_usage"),
            ("temperature", "soma.hardware.temperature"),
            ("thought_ms", "soma.latency.last_thought_ms"),
            ("perception_lag_ms", "soma.latency.perception_lag_ms"),
            ("token_velocity", "soma.latency.token_velocity"),
            ("pulse_rate", "soma.expressive.pulse_rate"),
            ("mycelium_density", "soma.expressive.mycelium_density"),
            ("vitality", "vitality"),
            ("sensor_load", "soma.sensors"),
        ),
    ),
    "A": _sch(
        "A",
        (
            ("valence", "affect.valence"),
            ("arousal", "affect.arousal"),
            ("curiosity", "affect.curiosity"),
            ("engagement", "affect.engagement"),
            ("social_hunger", "affect.social_hunger"),
            ("free_energy", "free_energy"),
            *((f"emotion_{name}", f"affect.emotions.{name}") for name in _EMOTIONS),
            ("heart_rate", "affect.physiology.heart_rate"),
            ("cortisol", "affect.physiology.cortisol"),
            ("adrenaline", "affect.physiology.adrenaline"),
            ("momentum", "affect.momentum"),
            ("action_urgency", "organ:free_energy.get_action_urgency"),
            ("surprise_trend", "organ:free_energy.get_trend"),
        ),
    ),
    "G": _sch(
        "G",
        (
            ("attention_present", "cognition.attention_focus"),
            ("attention_hash", "cognition.attention_focus"),
            ("coherence", "cognition.coherence_score"),
            ("fragmentation", "cognition.fragmentation_score"),
            ("contradictions", "cognition.contradiction_count"),
            ("conversation_energy", "cognition.conversation_energy"),
            ("discourse_depth", "cognition.discourse_depth"),
            ("branch_load", "cognition.discourse_branches"),
            ("phi", "phi"),
            ("selfhood_readings", "cognition.selfhood_reading"),
            ("ignition", "organ:workspace.ignition_level"),
            ("ignited", "organ:workspace.ignited"),
            ("candidates", "organ:workspace.pending_candidates"),
            ("winner_priority", "organ:workspace.last_priority"),
            ("winner_source", "organ:workspace.last_winner"),
            ("tick", "organ:workspace.tick"),
            ("tie_impasses", "organ:workspace.tie_impasses"),
            ("inhibited", "organ:workspace.inhibited_sources"),
            ("broadcasts", "organ:workspace.broadcast_history_len"),
            # The homeostatic modifiers: how hot, how deep, how creative and how
            # focused the next thought is allowed to be. They are the substrate
            # and the drives reaching cognitive control, which is the one place
            # the body's state becomes a parameter of thinking rather than a
            # sentence about it.
            ("modifier_temperature", "cognition.modifiers.temperature_mod"),
            ("modifier_depth", "cognition.modifiers.depth_mod"),
            ("modifier_creativity", "cognition.modifiers.creativity_mod"),
            ("modifier_focus", "cognition.modifiers.focus_mod"),
            ("modifier_vitality", "cognition.modifiers.overall_vitality"),
            ("modifier_urgency", "cognition.modifiers.urgency_flag"),
        ),
    ),
    "C": _sch(
        "C",
        (
            *((f"mode_{name}", "cognition.current_mode") for name in _MODES),
            ("loop_cycle", "loop_cycle"),
            ("phi_estimate", "phi_estimate"),
            ("phenomenal_valence", "cognition.phenomenal_state.valence"),
            ("phenomenal_arousal", "cognition.phenomenal_state.arousal"),
            ("phenomenal_coherence", "cognition.phenomenal_state.coherence"),
            ("phenomenal_energy", "cognition.phenomenal_state.energy"),
            # `cognition.phenomenal_state.latent_snapshot` is deliberately not
            # here. It is 128 numbers built by hashing the phenomenal claim, so
            # two nearby states produce unrelated vectors and the distance
            # between them means nothing; and grepping the runtime finds it
            # written once and read by nowhere, which fails the test at the top
            # of this file — a number nothing computes from is not state. In
            # the schema it contributed a third of a standard deviation to the
            # floor between two untouched runs and no signal at all.
            ("substrate_valence", "organ:substrate.valence"),
            ("substrate_arousal", "organ:substrate.arousal"),
            ("substrate_dominance", "organ:substrate.dominance"),
            ("substrate_energy", "organ:substrate.energy"),
            ("substrate_volatility", "organ:substrate.volatility"),
            ("substrate_focus", "organ:substrate.focus"),
            ("substrate_curiosity", "organ:substrate.curiosity"),
            ("substrate_frustration", "organ:substrate.frustration"),
            ("substrate_revision", "organ:substrate.state_revision"),
        ),
    ),
    "S": _sch(
        "S",
        (
            ("stability", "identity.stability"),
            ("evolution_score", "identity.evolution_score"),
            ("bonding_level", "identity.bonding_level"),
            ("narrative_version", "identity.narrative_version"),
            ("narrative_len", "identity.current_narrative"),
            ("value_load", "identity.core_values"),
            ("preference_load", "identity.self_preferences"),
            *(
                (f"trait_{trait}", f"identity.personality_growth.{trait}")
                for trait in (
                    "openness",
                    "conscientiousness",
                    "extraversion",
                    "agreeableness",
                    "neuroticism",
                )
            ),
            ("belief_count", "organ:self_model.belief_count"),
            ("belief_version", "organ:self_model.version"),
            ("snapshot_count", "organ:self_model.snapshot_count"),
            ("pending_updates", "organ:self_model.pending_update_count"),
            ("belief_digest", "organ:self_model.beliefs"),
            ("agency_acted", "organ:agency.acted"),
            ("agency_efficacy", "organ:agency.efficacy"),
            ("agency_authored_share", "organ:agency.authored_share"),
            ("agency_capabilities", "organ:agency.capabilities"),
            ("agency_last_actor", "organ:agency.last_actor"),
            # How well she predicts her own next internal state. The loop that
            # computes this runs every heartbeat and its surprise signal is
            # consumed downstream; the self-state schema was not reading the
            # one quantity most obviously about the self model's own accuracy.
            ("prediction_error", "organ:self_prediction.smoothed_error"),
            ("prediction_surprises", "organ:self_prediction.surprise_count"),
            ("valence_error", "organ:self_prediction.valence_error_ema"),
            ("drive_error", "organ:self_prediction.drive_error_ema"),
            ("focus_error", "organ:self_prediction.focus_error_ema"),
            ("least_predictable", "organ:self_prediction.most_unpredictable"),
            ("prediction_confidence", "organ:self_prediction.current_prediction.confidence"),
        ),
    ),
    "M": _sch(
        "M",
        (
            ("working_load", "cognition.working_memory"),
            ("working_recency", "cognition.working_memory[-1]"),
            ("retrieved_load", "cognition.long_term_memory"),
            ("summary_len", "cognition.rolling_summary"),
            ("ledger_load", "cognition.continuity_ledger"),
            ("thread_present", "cognition.active_thread_id"),
            ("cold_memory_load", "cold.long_term_memory"),
            ("evolution_log_load", "cold.evolution_log"),
        ),
    ),
    "W": _sch(
        "W",
        (
            ("entity_load", "world.known_entities"),
            ("relationship_load", "world.relationship_graph"),
            ("fact_load", "world.facts"),
            ("preference_load", "world.user_preferences"),
            ("fact_churn", "world.facts"),
            ("concept_load", "cold.concept_graph"),
            ("user_trend", "cognition.user_emotional_trend"),
            ("model_surprise", "organ:world_model.surprise"),
            ("model_facets", "organ:world_model.status"),
            ("model_observations", "organ:world_model.observations"),
        ),
    ),
    "D": _sch(
        "D",
        (
            ("goal_load", "cognition.active_goals"),
            ("initiative_load", "cognition.pending_initiatives"),
            ("goal_hash", "cognition.active_goals"),
            ("origin_is_user", "cognition.current_origin"),
            ("action_source_hash", "cognition.last_action_source"),
            *((f"drive_{name}", f"motivation.budgets.{name}") for name in _DRIVES),
        ),
    ),
    "N": _sch(
        "N",
        (
            ("novelty", "ontogeny.novelty"),
            ("displacement", "ontogeny.displacement"),
            ("steps", "ontogeny.steps"),
            ("era", "ontogeny.era"),
            ("hidden_norm", "ontogeny.h"),
            *((f"hidden_{index}", "ontogeny.h") for index in range(8)),
        ),
    ),
}


def schema(domain: str) -> Schema:
    return _SCHEMAS[domain]


def domain_width(domain: str) -> int:
    return _SCHEMAS[domain].width


def feature_names(domain: str | None = None) -> tuple[str, ...]:
    """Every column name, qualified by domain, in matrix order."""
    if domain is not None:
        return tuple(f"{domain}.{name}" for name in _SCHEMAS[domain].features)
    return tuple(
        f"{key}.{name}" for key in DOMAINS for name in _SCHEMAS[key].features
    )


TOTAL_WIDTH: int = sum(domain_width(key) for key in DOMAINS)


# ── reading ──────────────────────────────────────────────────────────────


def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _dig(root: Any, path: str, default: Any = None) -> Any:
    node = root
    for part in path.split("."):
        if node is None:
            return default
        if isinstance(node, Mapping):
            node = node.get(part, None)
        else:
            node = getattr(node, part, None)
    return default if node is None else node


def _sat(count: Any, scale: float) -> float:
    """A count on a bounded scale, so a list of 4000 does not swamp everything.

    x/(x+scale) rather than a clip, because a clip makes every busy state
    identical and the busy states are where the interesting differences live.
    """
    try:
        n = float(len(count))  # type: ignore[arg-type]
    except TypeError:
        n = _f(count)
    return n / (n + scale)


def _hash_unit(text: Any) -> float:
    """Content identity as a number in [0,1). Different text, different value.

    A hash carries no magnitude — the distance between two hashes means only
    "these differ" — which is what a change of topic or of objective is. The
    downstream measures treat it as any other coordinate, and the honest
    reading of a moved hash feature is that the content changed.
    """
    raw = str(text or "")
    if not raw:
        return 0.0
    digest = hashlib.blake2b(raw.encode("utf-8", "ignore"), digest_size=4).digest()
    return int.from_bytes(digest, "big") / float(1 << 32)


def _percept_novelty(percepts: Any) -> float:
    if not isinstance(percepts, list) or not percepts:
        return 0.0
    tail = percepts[-8:]
    seen = {_hash_unit(str(item)) for item in tail}
    return len(seen) / float(len(tail))


def _read_P(state: Any, now: float) -> np.ndarray:
    percepts = _dig(state, "world.recent_percepts", []) or []
    last = percepts[-1] if isinstance(percepts, list) and percepts else {}
    stamp = _f(_dig(last, "timestamp", 0.0)) if isinstance(last, Mapping) else 0.0
    recency = 0.0 if stamp <= 0 else 1.0 / (1.0 + max(0.0, now - stamp))
    sources = {
        str(_dig(item, "source", "?")) for item in percepts[-16:] if isinstance(item, Mapping)
    }
    objective = _dig(state, "cognition.current_objective", "") or ""
    return np.array(
        [
            _sat(percepts, 16.0),
            recency,
            _sat(sources, 4.0),
            _percept_novelty(percepts),
            1.0 if _dig(state, "world.spatial_context") else 0.0,
            _sat(_dig(state, "world.known_entities", {}) or {}, 8.0),
            _sat(str(objective), 64.0),
            _hash_unit(objective),
        ],
        dtype=np.float64,
    )


def _read_I(state: Any) -> np.ndarray:
    return np.array(
        [
            _f(_dig(state, "soma.hardware.cpu_usage")),
            _f(_dig(state, "soma.hardware.vram_usage")),
            _f(_dig(state, "soma.hardware.temperature")),
            _sat(_f(_dig(state, "soma.latency.last_thought_ms")), 2000.0),
            _sat(_f(_dig(state, "soma.latency.perception_lag_ms")), 500.0),
            _sat(_f(_dig(state, "soma.latency.token_velocity")), 40.0),
            _f(_dig(state, "soma.expressive.pulse_rate"), 1.0),
            _f(_dig(state, "soma.expressive.mycelium_density"), 0.5),
            _f(_dig(state, "vitality"), 1.0),
            _sat(_dig(state, "soma.sensors", {}) or {}, 4.0),
        ],
        dtype=np.float64,
    )


def _read_A(state: Any, organs: Organs) -> np.ndarray:
    emotions = _dig(state, "affect.emotions", {}) or {}
    physiology = _dig(state, "affect.physiology", {}) or {}
    head = [
        _f(_dig(state, "affect.valence")),
        _f(_dig(state, "affect.arousal"), 0.5),
        _f(_dig(state, "affect.curiosity"), 0.5),
        _f(_dig(state, "affect.engagement"), 0.5),
        _f(_dig(state, "affect.social_hunger"), 0.5),
        math.tanh(_f(_dig(state, "free_energy"))),
    ]
    head.extend(_f(emotions.get(name)) for name in _EMOTIONS)
    head.extend(
        [
            _sat(_f(physiology.get("heart_rate"), 72.0), 80.0),
            _sat(_f(physiology.get("cortisol"), 10.0), 20.0),
            _f(physiology.get("adrenaline")),
            _f(_dig(state, "affect.momentum"), 0.85),
            _f(_call(organs.free_energy, "get_action_urgency", 0.0)),
            _hash_unit(_call(organs.free_energy, "get_trend", "")),
        ]
    )
    return np.array(head, dtype=np.float64)


def _read_G(state: Any, organs: Organs) -> np.ndarray:
    focus = _dig(state, "cognition.attention_focus", "") or ""
    workspace = _call(organs.workspace, "get_status", {}) or {}
    modifiers = _dig(state, "cognition.modifiers", {}) or {}
    return np.array(
        [
            1.0 if focus else 0.0,
            _hash_unit(focus),
            _f(_dig(state, "cognition.coherence_score"), 1.0),
            _f(_dig(state, "cognition.fragmentation_score")),
            _sat(_f(_dig(state, "cognition.contradiction_count")), 4.0),
            _f(_dig(state, "cognition.conversation_energy"), 0.5),
            _sat(_f(_dig(state, "cognition.discourse_depth")), 8.0),
            _sat(_dig(state, "cognition.discourse_branches", []) or [], 4.0),
            math.tanh(_f(_dig(state, "phi"))),
            _sat(_dig(state, "cognition.selfhood_reading", {}) or {}, 4.0),
            _f(workspace.get("ignition_level")),
            1.0 if workspace.get("ignited") else 0.0,
            _sat(_f(workspace.get("pending_candidates")), 4.0),
            _f(workspace.get("last_priority")),
            _hash_unit(workspace.get("last_winner")),
            _sat(_f(workspace.get("tick")), 200.0),
            _sat(_f(workspace.get("tie_impasses")), 8.0),
            _sat(workspace.get("inhibited_sources") or [], 4.0),
            _sat(_f(workspace.get("broadcast_history_len")), 32.0),
            _f(modifiers.get("temperature_mod"), 1.0),
            _f(modifiers.get("depth_mod"), 1.0),
            _f(modifiers.get("creativity_mod"), 1.0),
            _f(modifiers.get("focus_mod"), 1.0),
            _f(modifiers.get("overall_vitality"), 1.0),
            1.0 if modifiers.get("urgency_flag") else 0.0,
        ],
        dtype=np.float64,
    )


def _latent(state: Any, width: int) -> list[float]:
    raw = _dig(state, "cognition.phenomenal_state.latent_snapshot", []) or []
    if not isinstance(raw, (list, tuple)) or not raw:
        return [0.0] * width
    values = np.asarray([_f(v) for v in raw], dtype=np.float64)
    if values.size < width:
        values = np.pad(values, (0, width - values.size))
    # Fixed-size summary by averaging equal blocks. A learned projection would
    # be a second model between the state and its measurement; block means are
    # a decision anyone can check by hand.
    blocks = np.array_split(values, width)
    return [float(np.mean(block)) if block.size else 0.0 for block in blocks]


def _read_C(state: Any, organs: Organs) -> np.ndarray:
    mode = _dig(state, "cognition.current_mode")
    label = str(getattr(mode, "value", mode) or "reactive").lower()
    head = [1.0 if label == name else 0.0 for name in _MODES]
    head.extend(
        [
            _sat(_f(_dig(state, "loop_cycle")), 100.0),
            math.tanh(_f(_dig(state, "phi_estimate"))),
            _f(_dig(state, "cognition.phenomenal_state.valence")),
            _f(_dig(state, "cognition.phenomenal_state.arousal")),
            _f(_dig(state, "cognition.phenomenal_state.coherence"), 1.0),
            _f(_dig(state, "cognition.phenomenal_state.energy")),
        ]
    )
    affect = _call(organs.substrate, "get_substrate_affect", {}) or {}
    status = _call(organs.substrate, "get_status", {}) or {}
    head.extend(
        [
            _f(affect.get("valence")),
            _f(affect.get("arousal")),
            _f(affect.get("dominance")),
            _f(affect.get("energy")),
            _f(affect.get("volatility")),
            _sat(_f(status.get("focus")), 50.0),
            _sat(_f(status.get("curiosity")), 50.0),
            _sat(_f(status.get("frustration")), 50.0),
            _sat(_f(status.get("state_revision")), 200.0),
        ]
    )
    return np.array(head, dtype=np.float64)


def _read_S(state: Any, organs: Organs) -> np.ndarray:
    growth = _dig(state, "identity.personality_growth", {}) or {}
    head = [
        _f(_dig(state, "identity.stability"), 1.0),
        _f(_dig(state, "identity.evolution_score")),
        _f(_dig(state, "identity.bonding_level")),
        _sat(_f(_dig(state, "identity.narrative_version")), 8.0),
        _sat(str(_dig(state, "identity.current_narrative", "") or ""), 512.0),
        _sat(_dig(state, "identity.core_values", []) or [], 8.0),
        _sat(_dig(state, "identity.self_preferences", {}) or {}, 8.0),
    ]
    head.extend(
        _f(growth.get(trait))
        for trait in (
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
        )
    )
    introspection = _call(organs.self_model, "get_introspection", {}) or {}
    beliefs = getattr(organs.self_model, "beliefs", {}) or {}
    agency = _call(organs.agency, "snapshot", {}) or {}
    prediction = _call(organs.self_prediction, "get_snapshot", {}) or {}
    current = prediction.get("current_prediction") or {}
    head.extend(
        [
            _sat(_f(introspection.get("belief_count")), 16.0),
            _sat(_f(introspection.get("version")), 32.0),
            _sat(_f(introspection.get("snapshot_count")), 8.0),
            _sat(_f(introspection.get("pending_update_count")), 4.0),
            _hash_unit(",".join(f"{k}={beliefs[k]}" for k in sorted(beliefs)[:16])),
            _sat(_f(agency.get("acted")), 16.0),
            _f(agency.get("efficacy")),
            _f(agency.get("authored_share")),
            _sat(_f(agency.get("capabilities")), 8.0),
            _hash_unit(agency.get("last_actor", "")),
            _f(prediction.get("smoothed_error")),
            _sat(_f(prediction.get("surprise_count")), 16.0),
            _f(prediction.get("valence_error_ema")),
            _f(prediction.get("drive_error_ema")),
            _f(prediction.get("focus_error_ema")),
            _hash_unit(prediction.get("most_unpredictable", "")),
            _f(current.get("confidence")),
        ]
    )
    return np.array(head, dtype=np.float64)


def _read_M(state: Any) -> np.ndarray:
    working = _dig(state, "cognition.working_memory", []) or []
    last = working[-1] if isinstance(working, list) and working else {}
    return np.array(
        [
            _sat(working, 24.0),
            _hash_unit(str(last)),
            _sat(_dig(state, "cognition.long_term_memory", []) or [], 8.0),
            _sat(str(_dig(state, "cognition.rolling_summary", "") or ""), 512.0),
            _sat(_dig(state, "cognition.continuity_ledger", {}) or {}, 8.0),
            1.0 if _dig(state, "cognition.active_thread_id") else 0.0,
            _sat(_dig(state, "cold.long_term_memory", []) or [], 64.0),
            _sat(_dig(state, "cold.evolution_log", []) or [], 32.0),
        ],
        dtype=np.float64,
    )


def _read_W(state: Any, organs: Organs) -> np.ndarray:
    facts = _dig(state, "world.facts", {}) or {}
    status = _call(organs.world_model, "status", {}) or {}
    facets = status.get("facets", {}) if isinstance(status, Mapping) else {}
    surprise = _call(organs.world_model, "surprise", None)
    return np.array(
        [
            _sat(_dig(state, "world.known_entities", {}) or {}, 8.0),
            _sat(_dig(state, "world.relationship_graph", {}) or {}, 8.0),
            _sat(facts, 16.0),
            _sat(_dig(state, "world.user_preferences", {}) or {}, 8.0),
            _hash_unit(",".join(sorted(str(k) for k in facts)[:32])),
            _sat(_dig(state, "cold.concept_graph", {}) or {}, 32.0),
            _hash_unit(_dig(state, "cognition.user_emotional_trend", "neutral")),
            math.tanh(_f(surprise)) if surprise is not None else 0.0,
            _sat(
                [name for name, entry in facets.items() if entry.get("available")], 3.0
            )
            if isinstance(facets, Mapping)
            else 0.0,
            _f(getattr(organs.world_model, "_observations", 0.0)),
        ],
        dtype=np.float64,
    )


def _read_D(state: Any) -> np.ndarray:
    goals = _dig(state, "cognition.active_goals", []) or []
    budgets = _dig(state, "motivation.budgets", {}) or {}
    head = [
        _sat(goals, 8.0),
        _sat(_dig(state, "cognition.pending_initiatives", []) or [], 4.0),
        _hash_unit(str(goals[:3])),
        1.0 if str(_dig(state, "cognition.current_origin", "")).startswith("user") else 0.0,
        _hash_unit(_dig(state, "cognition.last_action_source", "")),
    ]
    for name in _DRIVES:
        entry = budgets.get(name)
        if isinstance(entry, Mapping):
            head.append(_f(entry.get("current", entry.get("level", 0.0))))
        else:
            head.append(_f(entry))
    return np.array(head, dtype=np.float64)


def _read_N(ontogeny: Any) -> np.ndarray:
    if ontogeny is None:
        return np.zeros(domain_width("N"), dtype=np.float64)
    hidden = np.asarray(getattr(ontogeny, "h", np.zeros(1)), dtype=np.float64).reshape(-1)
    blocks = np.array_split(hidden, 8) if hidden.size else [np.zeros(1)] * 8
    head = [
        _f(getattr(ontogeny, "last_novelty", 0.5), 0.5),
        _f(getattr(ontogeny, "last_displacement", 0.0)),
        _sat(_f(getattr(ontogeny, "steps", 0.0)), 200.0),
        _sat(_f(getattr(ontogeny, "era", 1.0)), 4.0),
        float(np.linalg.norm(hidden) / max(1.0, math.sqrt(hidden.size))),
    ]
    head.extend(float(np.mean(block)) if block.size else 0.0 for block in blocks)
    return np.array(head, dtype=np.float64)


#: Readers that need the organs as well as the state.
_ORGAN_READERS: dict[str, Callable[[Any, Organs], np.ndarray]] = {
    "A": _read_A,
    "G": _read_G,
    "C": _read_C,
    "S": _read_S,
    "W": _read_W,
}

#: Readers that only need the state object.
_STATE_READERS: dict[str, Callable[[Any], np.ndarray]] = {
    "I": _read_I,
    "M": _read_M,
    "D": _read_D,
}


@dataclass(frozen=True, slots=True)
class CoreState:
    """K_t, as ten named vectors and the concatenation of them."""

    values: dict[str, np.ndarray]
    t: float
    condition: str = ""
    tag: str = ""
    env: dict[str, float] = field(default_factory=dict)

    def vector(self) -> np.ndarray:
        return np.concatenate([self.values[key] for key in DOMAINS])

    def domain(self, key: str) -> np.ndarray:
        return self.values[key]

    def env_vector(self) -> np.ndarray:
        keys = sorted(self.env)
        return np.array([self.env[key] for key in keys], dtype=np.float64)


def domain_slices() -> dict[str, slice]:
    out: dict[str, slice] = {}
    start = 0
    for key in DOMAINS:
        width = domain_width(key)
        out[key] = slice(start, start + width)
        start += width
    return out


def read_core_state(
    state: Any,
    *,
    ontogeny: Any = None,
    organs: Organs | None = None,
    condition: str = "",
    tag: str = "",
    env: Mapping[str, float] | None = None,
    now: float | None = None,
) -> CoreState:
    """One reading of the ten domains off live objects.

    ``state`` is an ``AuraState``, ``ontogeny`` is the lifetime reservoir and
    ``organs`` are the live workspace, substrate, free-energy engine, self model
    and world model. Anything missing reads as zeros, which is a flat column
    rather than a guess at what it would have said.
    """
    moment = time.time() if now is None else now
    kit = organs or Organs()
    values = {"P": _read_P(state, moment), "N": _read_N(ontogeny)}
    for key, reader in _STATE_READERS.items():
        values[key] = reader(state)
    for key, organ_reader in _ORGAN_READERS.items():
        values[key] = organ_reader(state, kit)
    for key in DOMAINS:
        width = domain_width(key)
        got = values[key]
        if got.size != width:  # a reader and its schema must agree
            raise ValueError(f"domain {key} read {got.size} features, schema says {width}")
    return CoreState(
        values=values,
        t=moment,
        condition=condition,
        tag=tag,
        env=dict(env or {}),
    )


# ── writing ──────────────────────────────────────────────────────────────
#
# do(X_i = x_i + delta). Each writer touches the same attributes its reader
# reads, so an intervention that shows nothing is a fact about the coupling
# and not about the instrument. `perturbable` names the domains that have a
# writer at all, and the battery refuses to score a domain that has none.


def _bump(obj: Any, path: str, delta: float, lo: float, hi: float) -> bool:
    parts = path.split(".")
    node = obj
    for part in parts[:-1]:
        node = node.get(part) if isinstance(node, Mapping) else getattr(node, part, None)
        if node is None:
            return False
    name = parts[-1]
    if isinstance(node, dict):
        current = _f(node.get(name))
        node[name] = min(hi, max(lo, current + delta))
        return True
    if not hasattr(node, name):
        return False
    current = _f(getattr(node, name))
    setattr(node, name, min(hi, max(lo, current + delta)))
    return True


def _perturb_P(state: Any, delta: float, ontogeny: Any) -> bool:
    del ontogeny
    percepts = _dig(state, "world.recent_percepts", None)
    if not isinstance(percepts, list):
        return False
    percepts.append(
        {
            "source": "subject_core_probe",
            "content": f"probe delta {delta:+.4f}",
            "timestamp": time.time(),
            "salience": abs(delta),
        }
    )
    return True


def _perturb_I(state: Any, delta: float, ontogeny: Any) -> bool:
    del ontogeny
    # These fields are percentages and milliseconds, not fractions. The first
    # version clamped cpu_usage to [0, 1] while the runtime writes 0..100, so
    # displacing the body by +0.15 set it to one percent — a large move in the
    # wrong direction, dressed as a small one in the right one.
    hit = _bump(state, "soma.hardware.temperature", delta * 20.0, 0.0, 110.0)
    hit |= _bump(state, "soma.hardware.cpu_usage", delta * 100.0, 0.0, 100.0)
    hit |= _bump(state, "soma.hardware.vram_usage", delta * 100.0, 0.0, 100.0)
    hit |= _bump(state, "soma.hardware.ram_usage", delta * 100.0, 0.0, 100.0)
    hit |= _bump(state, "soma.latency.last_thought_ms", delta * 500.0, 0.0, 60_000.0)
    hit |= _bump(state, "vitality", -abs(delta), 0.0, 1.0)
    return hit


def _perturb_A(state: Any, delta: float, ontogeny: Any) -> bool:
    del ontogeny
    hit = _bump(state, "affect.valence", delta, -1.0, 1.0)
    hit |= _bump(state, "affect.arousal", delta, 0.0, 1.0)
    hit |= _bump(state, "affect.curiosity", delta, 0.0, 1.0)
    return hit


def _perturb_G(state: Any, delta: float, ontogeny: Any) -> bool:
    del ontogeny
    hit = _bump(state, "cognition.conversation_energy", delta, 0.0, 1.0)
    hit |= _bump(state, "cognition.coherence_score", -abs(delta), 0.0, 1.0)
    node = getattr(state, "cognition", None)
    if node is not None:
        node.attention_focus = f"probe:{delta:+.4f}"
        hit = True
    return hit


def _perturb_C(state: Any, delta: float, ontogeny: Any) -> bool:
    del ontogeny
    hit = _bump(state, "phi_estimate", delta, -10.0, 10.0)
    node = _dig(state, "cognition.phenomenal_state", None)
    if node is not None and hasattr(node, "valence"):
        node.valence = min(1.0, max(-1.0, _f(node.valence) + delta))
        node.arousal = min(1.0, max(0.0, _f(node.arousal) + delta))
        snapshot = list(getattr(node, "latent_snapshot", []) or [])
        if snapshot:
            node.latent_snapshot = [
                min(1.0, max(-1.0, _f(v) + delta)) for v in snapshot
            ]
        hit = True
    return hit


def _perturb_S(state: Any, delta: float, ontogeny: Any) -> bool:
    del ontogeny
    hit = _bump(state, "identity.stability", -abs(delta), 0.0, 1.0)
    hit |= _bump(state, "identity.bonding_level", delta, 0.0, 1.0)
    hit |= _bump(state, "identity.evolution_score", delta, -10.0, 10.0)
    growth = _dig(state, "identity.personality_growth", None)
    if isinstance(growth, dict):
        growth["openness"] = _f(growth.get("openness")) + delta
        hit = True
    return hit


def _perturb_M(state: Any, delta: float, ontogeny: Any) -> bool:
    del ontogeny
    working = _dig(state, "cognition.working_memory", None)
    if not isinstance(working, list):
        return False
    working.append(
        {
            "role": "probe",
            "content": f"subject-core memory probe {delta:+.4f}",
            "timestamp": time.time(),
        }
    )
    return True


def _perturb_W(state: Any, delta: float, ontogeny: Any) -> bool:
    del ontogeny
    facts = _dig(state, "world.facts", None)
    if not isinstance(facts, dict):
        return False
    facts["subject_core_probe"] = {"delta": delta, "at": time.time()}
    return True


def _perturb_D(state: Any, delta: float, ontogeny: Any) -> bool:
    del ontogeny
    goals = _dig(state, "cognition.active_goals", None)
    hit = False
    if isinstance(goals, list):
        goals.append(
            {
                "id": "subject_core_probe",
                "goal": f"probe intention {delta:+.4f}",
                "origin": "probe",
                "status": "pending",
            }
        )
        hit = True
    budgets = _dig(state, "motivation.budgets", None)
    if isinstance(budgets, dict):
        for name in _DRIVES:
            entry = budgets.get(name)
            if isinstance(entry, dict):
                for key in ("current", "level"):
                    if key in entry:
                        entry[key] = _f(entry[key]) + delta
                        hit = True
    return hit


def _perturb_N(state: Any, delta: float, ontogeny: Any) -> bool:
    del state
    if ontogeny is None or not hasattr(ontogeny, "h"):
        return False
    hidden = np.asarray(ontogeny.h, dtype=np.float64)
    ontogeny.h = np.clip(hidden + delta, -1.0, 1.0)
    return True


_WRITERS: dict[str, Callable[[Any, float, Any], bool]] = {
    "P": _perturb_P,
    "I": _perturb_I,
    "A": _perturb_A,
    "G": _perturb_G,
    "C": _perturb_C,
    "S": _perturb_S,
    "M": _perturb_M,
    "W": _perturb_W,
    "D": _perturb_D,
    "N": _perturb_N,
}


def perturbable() -> tuple[str, ...]:
    """The domains an intervention can actually reach."""
    return tuple(key for key in DOMAINS if key in _WRITERS)


def perturb(state: Any, domain: str, delta: float, *, ontogeny: Any = None) -> bool:
    """Displace one domain by delta. False means the write found nothing.

    A False is the useful answer: it says this domain is not writable in this
    runtime, which makes every edge out of it unmeasured rather than absent.
    """
    writer = _WRITERS.get(domain)
    if writer is None:
        return False
    return bool(writer(state, float(delta), ontogeny))


async def perturb_organs(organs: Organs, domain: str, delta: float) -> bool:
    """Displace the part of a domain that lives in an organ rather than in state.

    Each write here goes through the organ's own public path — the substrate's
    gated update, the world model's observe, the self model's belief update —
    rather than reaching past it into an attribute. A perturbation that a
    subsystem's own authority would refuse is not a perturbation of that
    subsystem, and forcing it would measure a state the runtime can never
    reach.
    """
    hit = False
    if domain == "C" and organs.substrate is not None:
        try:
            await organs.substrate.update(
                delta_frustration=delta, delta_curiosity=delta, source="subject_core_probe"
            )
            hit = True
        except Exception:  # noqa: BLE001 - a refused write is not a write
            hit = False
    elif domain == "G" and organs.workspace is not None:
        workspace = organs.workspace
        # Displacing the workspace means changing what wins, not nudging a
        # readout. Writing `ignition_level` moved the number the schema reads
        # and nothing downstream, because the consumers fire on a broadcast and
        # a broadcast comes from a competition. This enters a bid strong enough
        # to change the outcome, which is the workspace intervention the
        # specification asks for: perturb one workspace content.
        try:
            from core.consciousness.global_workspace import CognitiveCandidate, ContentType

            await workspace.submit(
                CognitiveCandidate(
                    content=f"subject core probe {delta:+.4f}",
                    source="subject_core_probe",
                    priority=min(1.0, max(0.0, 0.5 + delta * 3.0)),
                    content_type=ContentType.META,
                    affect_weight=abs(delta),
                )
            )
            hit = True
        except Exception:  # noqa: BLE001
            hit = False
    elif domain == "S" and organs.self_model is not None:
        try:
            await organs.self_model.update_belief(
                "subject_core_probe", round(delta, 4), note="displacement probe"
            )
            hit = True
        except Exception:  # noqa: BLE001
            hit = False
    elif domain == "W" and organs.world_model is not None:
        try:
            organs.world_model.observe(
                {"subject_core_probe": delta, "at": time.time()}, learn=True
            )
            hit = True
        except Exception:  # noqa: BLE001
            hit = False
    return hit
