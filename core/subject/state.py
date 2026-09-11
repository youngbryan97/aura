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

import contextlib
import hashlib
import math
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.state.percepts import emit_percept, read_percept

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
    comparator: Any = None
    #: The resilience engine. Carried because the body it holds is the body
    #: homeostasis reads, and an arm that displaces interoception has to
    #: displace that one too or the displacement stops at the state.
    soma: Any = None

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
            comparator=_agency_comparator(),
            soma=service("soma"),
        )


def _agency_comparator() -> Any:
    """The efference-copy comparator, if it is there."""
    try:
        from core.consciousness.agency_comparator import get_agency_comparator

        return get_agency_comparator()
    except Exception:  # noqa: BLE001 - an absent comparator is an absent organ
        return None


#: What went wrong while the current reading was taken. A reader that failed
#: and a genuine zero are different states, and until this existed they were
#: the same number: an absent organ, a raising reader and a real reading of
#: zero all reached the recording as 0.0, so a criterion could fail on a
#: subsystem that was never asked. The reading records the misses beside the
#: values and the run says whether it can be read at all.
_MISSES: ContextVar[dict[str, str] | None] = ContextVar("subject_core_misses", default=None)


def _miss(source: str, reason: str) -> None:
    record = _MISSES.get()
    if record is not None:
        record.setdefault(source, reason)


@contextlib.contextmanager
def recording_misses() -> Iterator[dict[str, str]]:
    """Collect what could not be read while this block runs."""
    record: dict[str, str] = {}
    token = _MISSES.set(record)
    try:
        yield record
    finally:
        _MISSES.reset(token)


def _call(obj: Any, name: str, default: Any = None, *, source: str = "") -> Any:
    """Call a reader on an organ, and say so when there was no reading."""
    if obj is None:
        _miss(source or name, "organ absent")
        return default
    method = getattr(obj, name, None)
    if method is None:
        _miss(source or name, "reader absent")
        return default
    try:
        return method()
    except Exception as exc:  # noqa: BLE001 - an organ that raises has told us nothing
        _miss(source or name, f"reader raised {type(exc).__name__}")
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
#: Read from the state's own defaults rather than named here. The first
#: version listed `rest` and `creation`, which no budget has ever been called —
#: so two of deliberation's columns were a constant zero, and the probe that
#: displaces this domain moved neither `growth`, the drive that is most
#: depleted on almost every tick and therefore decides what she does, nor
#: `integrity` or `energy`. A displacement that cannot reach the drive that
#: decides is a displacement of deliberation that deliberation cannot notice.
def _budget_names() -> tuple[str, ...]:
    try:
        from core.motivation.constants import MOTIVATION_BUDGET_DEFAULTS

        return tuple(sorted(MOTIVATION_BUDGET_DEFAULTS))
    except ImportError:  # pragma: no cover - the constants are not optional
        return ("curiosity", "energy", "growth", "integrity", "social")


_DRIVES: tuple[str, ...] = _budget_names()

#: Cognitive modes, one-hot into C.
_MODES: tuple[str, ...] = ("reactive", "deliberate", "dreaming", "dormant")

#: How many buckets a body of content is spread across. Four is enough for two
#: recollections that share most of their words to land near each other and two
#: that share none to land apart, and few enough not to crowd the schema.
CONTENT_BUCKETS: int = 4

#: Vocabularies fixed before the run, so a category is a set of coordinates
#: rather than a hash magnitude. Each carries one extra slot for a value
#: outside the list, because a vocabulary that silently drops what it has not
#: seen reports an unfamiliar state as the absence of a state.
_UNPREDICTABLE_DIMENSIONS: tuple[str, ...] = (
    "affect_valence",
    "drive_state",
    "attentional_focus",
)

#: Who an event is attributed to. `core.agency.authorship.SELF` is "self";
#: everything else the ledger records names the party that caused it.
_ACTORS: tuple[str, ...] = ("self", "user", "world", "system")

#: An ordered ladder is not a set of names. Coding it one-hot throws away the
#: order — the distance from "strongly self-authored" to "mixed" would equal
#: the distance to "mostly world-caused" — so these are read as a position on
#: the ladder, with a separate column saying whether there was a reading at
#: all. Absent and middling are different states and a single scalar cannot
#: hold both.
_ATTRIBUTION_LADDER: dict[str, float] = {
    "mostly world-caused": 0.0,
    "mixed attribution": 0.5,
    "mostly self-authored": 0.75,
    "strongly self-authored": 1.0,
}

_USER_TREND_LADDER: dict[str, float] = {
    "cooling_off": 0.0,
    "neutral": 0.5,
    "warming_up": 0.75,
    "engaged": 1.0,
}

_FREE_ENERGY_TREND_LADDER: dict[str, float] = {
    "falling": 0.0,
    "stable": 0.5,
    "rising": 1.0,
}


_SCHEMAS: dict[str, Schema] = {
    "P": _sch(
        "P",
        (
            ("percept_load", "world.recent_percepts"),
            ("percept_claim", "world.recent_percepts[*].salience"),
            ("percept_sources", "world.recent_percepts[*].type"),
            ("percept_unfelt", "world.recent_percepts[*].consumed_by"),
            ("percept_strength", "world.recent_percepts[*].intensity"),
            ("spatial_present", "world.spatial_context"),
            ("objective_len", "cognition.current_objective"),
            *[
                (f"objective_profile_{i}", "cognition.current_objective")
                for i in range(CONTENT_BUCKETS)
            ],
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
            # Her own exertion, which is not the machine's load. The host
            # readings above are the environment; this is what she spent
            # thinking, and it is the one body channel an experiment can hold
            # the host still without also holding still.
            ("exertion", "soma.exertion"),
            ("recall_effort", "soma.effort"),
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
            ("surprise_trend_known", "organ:free_energy.get_trend"),
        ),
    ),
    "G": _sch(
        "G",
        (
            ("attention_present", "cognition.attention_focus"),
            *[
                (f"attention_profile_{i}", "cognition.attention_focus")
                for i in range(CONTENT_BUCKETS)
            ],
            ("coherence", "cognition.coherence_score"),
            ("fragmentation", "cognition.fragmentation_score"),
            ("contradictions", "cognition.contradiction_count"),
            ("conversation_energy", "cognition.conversation_energy"),
            ("discourse_depth", "cognition.discourse_depth"),
            ("branch_load", "cognition.discourse_branches"),
            # `phi` is not here. Executive closure assigns `state.phi` and
            # `state.phi_estimate` the same number in the same statement, and
            # the second of them is recurrent cognition's column — so the two
            # domains were reading one variable and any displacement of either
            # arrived in both with nothing in between. Integrated information
            # is a property of the recurrent system, not of what has attention,
            # so it stays with C and leaves here.
            ("selfhood_readings", "cognition.selfhood_reading"),
            ("ignition", "organ:workspace.ignition_level"),
            ("ignited", "organ:workspace.ignited"),
            ("candidates", "organ:workspace.pending_candidates"),
            ("winner_priority", "organ:workspace.last_priority"),
            *[
                (f"winner_source_{i}", "organ:workspace.last_winner")
                for i in range(CONTENT_BUCKETS)
            ],
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
            # The phenomenal field's four numbers are deliberately not here,
            # and neither is its latent snapshot.
            #
            # `make_phenomenal_field` builds valence and arousal from
            # `affect.*`, energy from the energy budget and coherence from
            # `cognition.coherence_score`. They are copies, made once a turn,
            # of three other domains' state, and grepping the runtime finds
            # nothing that reads any of them — only the claim, and whether a
            # field is there at all. So they failed the test at the top of this
            # file twice over: a number nothing computes from is not state, and
            # a verbatim copy of another domain's column is that domain being
            # read through this one. Four of recurrent cognition's nineteen
            # features were affect, deliberation and attention wearing its
            # name, and every edge into C was partly that copy arriving.
            #
            # The snapshot went for the first of those reasons alone: 128
            # numbers built by hashing the claim, so two nearby states produce
            # unrelated vectors and the distance between them means nothing. In
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
            *[
                (f"belief_profile_{i}", "organ:self_model.beliefs")
                for i in range(CONTENT_BUCKETS)
            ],
            ("agency_acted", "organ:agency.acted"),
            ("agency_efficacy", "organ:agency.efficacy"),
            ("agency_authored_share", "organ:agency.authored_share"),
            ("agency_capabilities", "organ:agency.capabilities"),
            *[
                (f"agency_actor_{name}", "organ:agency.last_actor")
                for name in (*_ACTORS, "other")
            ],
            # How well she predicts her own next internal state. The loop that
            # computes this runs every heartbeat and its surprise signal is
            # consumed downstream; the self-state schema was not reading the
            # one quantity most obviously about the self model's own accuracy.
            ("prediction_error", "organ:self_prediction.smoothed_error"),
            ("prediction_surprises", "organ:self_prediction.surprise_count"),
            ("valence_error", "organ:self_prediction.valence_error_ema"),
            ("drive_error", "organ:self_prediction.drive_error_ema"),
            ("focus_error", "organ:self_prediction.focus_error_ema"),
            *[
                (f"least_predictable_{name}", "organ:self_prediction.most_unpredictable")
                for name in (*_UNPREDICTABLE_DIMENSIONS, "other")
            ],
            ("prediction_confidence", "organ:self_prediction.current_prediction.confidence"),
            # The efference-copy comparator: how much of what happened her own
            # action explains. It emits and compares — both of those are wired
            # — and every one of its readouts was called from nowhere, so the
            # sense of agency it computes reached no part of her.
            ("agency_score", "organ:comparator.agency_score"),
            ("agency_traces", "organ:comparator.total_traces"),
            ("agency_pending", "organ:comparator.pending_efferences"),
            ("agency_attribution", "organ:comparator.recent_attribution"),
            ("agency_attribution_known", "organ:comparator.recent_attribution"),
        ),
    ),
    "M": _sch(
        "M",
        (
            ("working_load", "cognition.working_memory"),
            ("working_recency", "cognition.working_memory[-1]"),
            ("retrieved_load", "cognition.long_term_memory"),
            # What is in mind, not how much of it. The retrieved set is bounded
            # and fills within a few turns, so its length is constant from then
            # on while its contents change every cycle — an active memory read
            # as a count is a domain nothing can be shown to reach.
            *[
                (f"retrieved_profile_{i}", "cognition.long_term_memory[*]")
                for i in range(CONTENT_BUCKETS)
            ],
            # How strongly what is in mind was recalled. Retrieval ranks by
            # this and the number is what the workspace prices a recollection
            # by, so it is part of active memory's state rather than a
            # bookkeeping detail of the phase that produced it.
            ("recall_score", "cognition.memory_scores"),
            *[
                (f"working_profile_{i}", "cognition.working_memory[*]")
                for i in range(CONTENT_BUCKETS)
            ],
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
            *[
                (f"fact_profile_{i}", "world.facts")
                for i in range(CONTENT_BUCKETS)
            ],
            ("concept_load", "cold.concept_graph"),
            ("user_trend", "cognition.user_emotional_trend"),
            ("user_trend_known", "cognition.user_emotional_trend"),
            ("model_surprise", "organ:world_model.surprise"),
            # The learned world model carries a latent it steps on every
            # observation, and a running surprise. Three counts of dictionary
            # entries were most of what W read before this, which is why the
            # partition search kept finding it cheap to cut off: a domain read
            # as three slowly-moving counters is a domain that predicts itself.
            ("model_hidden_norm", "organ:world_model.learned.hidden_norm"),
            ("model_mean_surprise", "organ:world_model.learned.mean_surprise"),
            ("model_last_surprise", "organ:world_model.learned.last_surprise"),
            ("model_steps", "organ:world_model.learned.step_count"),
            ("causal_nodes", "organ:world_model.causal.nodes"),
            ("causal_edges", "organ:world_model.causal.edges"),
            ("causal_confirmed", "organ:world_model.causal.causal_edges"),
            ("model_facets", "organ:world_model.status"),
            ("model_train_steps", "organ:world_model.learned.train_steps"),
        ),
    ),
    "D": _sch(
        "D",
        (
            ("goal_load", "cognition.active_goals"),
            ("initiative_load", "cognition.pending_initiatives"),
            # How hard the open intentions are pressing, which is the quantity
            # every other domain moves and the one this domain had no column
            # for. The load is a count: an intention generated on every turn
            # left it flat while its urgency tracked the coherence, the
            # surprise, the distress, the strain and the novelty of the moment,
            # and none of that could reach deliberation's own state.
            ("initiative_urgency", "cognition.pending_initiatives[*].urgency"),
            ("goal_urgency", "cognition.active_goals[*].urgency"),
            *[
                (f"goal_profile_{i}", "cognition.active_goals")
                for i in range(CONTENT_BUCKETS)
            ],
            ("origin_is_user", "cognition.current_origin"),
            *[
                (f"action_source_{i}", "cognition.last_action_source")
                for i in range(CONTENT_BUCKETS)
            ],
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
            _miss(path, "path absent")
            return default
        if isinstance(node, Mapping):
            node = node.get(part, None)
        else:
            node = getattr(node, part, None)
    if node is None:
        _miss(path, "value absent")
        return default
    return node


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

    No column reads one any more. Eleven did, and every one of them was a
    coordinate in a space where the distance between two values meant nothing:
    two adjacent categories sat as far apart as any other pair, and the one on
    active memory's newest item hashed a dictionary that carries the instant it
    arrived, so the column was a clock. Categories are one-hot now, ordered
    ones are a position on their ladder, and bodies of text are
    `_content_buckets`. What is left here is counting distinct items, where a
    hash is exactly right because only equality is asked of it.
    """
    raw = str(text or "")
    if not raw:
        return 0.0
    digest = hashlib.blake2b(raw.encode("utf-8", "ignore"), digest_size=4).digest()
    return int.from_bytes(digest, "big") / float(1 << 32)


#: How many tokens enter the profile. Bounded so a long document does not cost
#: more to read than a short one.
_CONTENT_TOKENS: int = 64



#: Keys whose value is bookkeeping rather than content: when a thing was
#: written, which run wrote it, what its identifier is. A profile built from
#: `str(entry)` reads all of them as words, so the clock a working-memory entry
#: carries became four of active memory's coordinates — and two arms of one
#: trial, appending the same sentence a second apart, differed across the whole
#: profile before either had been displaced.
_BOOKKEEPING_KEYS: frozenset[str] = frozenset(
    {
        "timestamp",
        "at",
        "time",
        "created_at",
        "written_at",
        "id",
        "uuid",
        "turn",
        "run_id",
        "runtime_instance_id",
        "state_digest",
        "digest",
        "hash",
        "seq",
        "index",
    }
)


def _content_of(entry: Any, limit: int = 240) -> str:
    """The part of an entry a profile should be built from.

    A record is content plus bookkeeping. The content is what a later moment
    could recall; the bookkeeping is when it was stored and by whom, and it
    changes on every write whatever the content was.
    """
    if isinstance(entry, Mapping):
        parts = []
        for key in sorted(entry):
            if str(key).lower() in _BOOKKEEPING_KEYS:
                continue
            value = entry[key]
            if isinstance(value, Mapping):
                value = _content_of(value, limit)
            parts.append(f"{key}={value}")
        return " ".join(parts)[:limit]
    return str(entry)[:limit]


def _content_buckets(text: Any, buckets: int = CONTENT_BUCKETS) -> list[float]:
    """A body of content as a coordinate: sharing most words means being close.

    These fields were one hash of the whole string. A hash has no magnitude —
    two recollections differing by a word are as far apart as two with nothing
    in common — and every measure downstream is a distance. So one word
    changing in what she had in mind moved active memory's coordinate by a full
    standard deviation, which put that whole scale into the floor two identical
    sham arms could not get below, and out of reach of any intervention.

    Each token is projected onto every coordinate, by a fixed function of the
    token itself, and the profile is the mean over tokens. Two contents sharing
    most of their tokens have nearly the same profile; two sharing none are
    near-orthogonal, and the inner product between two profiles rises with the
    share of tokens they have in common. Every coordinate carries every token,
    which is what the first version did not do: counting tokens into buckets
    left a one-word field with three coordinates at zero whatever it said, and
    a coordinate that cannot move is not a measurement.
    """
    raw = str(text or "")
    if not raw:
        return [0.0] * buckets
    tokens = raw.split() or [raw]
    kept = tokens[:_CONTENT_TOKENS]
    totals = [0.0] * buckets
    for token in kept:
        digest = hashlib.blake2b(
            token.encode("utf-8", "ignore"), digest_size=2 * buckets
        ).digest()
        for index in range(buckets):
            word = int.from_bytes(digest[2 * index : 2 * index + 2], "big")
            totals[index] += word / 32767.5 - 1.0
    count = float(len(kept))
    return [value / count for value in totals]


def _one_hot(value: Any, vocabulary: Sequence[str]) -> list[float]:
    """A category as one column per name, plus one for everything else.

    A hash of a label has a magnitude nobody meant: two categories are as far
    apart as the arithmetic of their digests happens to make them, and every
    measure downstream is a distance. One column per name gives every pair of
    distinct categories the same distance, which is what "different category"
    means.
    """
    label = str(value or "").strip()
    out = [0.0] * (len(vocabulary) + 1)
    for index, name in enumerate(vocabulary):
        if label == name:
            out[index] = 1.0
            return out
    out[-1] = 1.0 if label else 0.0
    return out


def _ladder(value: Any, rungs: Mapping[str, float]) -> list[float]:
    """A position on an ordered ladder, and whether there was a reading.

    Two columns: where on the ladder, and known or not. Without the second, an
    absent reading has to be spelled as some position on the ladder, and every
    choice of position is a claim the instrument did not measure.
    """
    label = str(value or "").strip()
    if label in rungs:
        return [float(rungs[label]), 1.0]
    return [0.0, 0.0]


def _age(stamp: Any, *, span: float, now: float | None = None) -> float:
    """How long ago, as a share of `span`, saturating at one.

    A recency read as a hash of the item's text is not a recency; it is the
    item's identity, and where the text carries the instant it arrived, it is
    the clock. Both were true of active memory's second column.
    """
    try:
        value = float(stamp)
    except (TypeError, ValueError):
        return 1.0
    if value <= 0.0:
        return 1.0
    reference = time.time() if now is None else now
    return min(1.0, max(0.0, (reference - value) / span))


#: What counts as recent for a working-memory item, in seconds. A turn of the
#: offline organism takes a few seconds and a conversation runs for minutes, so
#: five minutes is the span over which "just now" and "a while ago" separate.
_WORKING_MEMORY_SPAN: float = 300.0


def _unfelt_share(percepts: Any) -> float:
    """The share of the stream that affect has not taken yet."""
    if not isinstance(percepts, list) or not percepts:
        return 0.0
    from core.state.percepts import fresh_for

    return len(fresh_for(percepts, "affect")) / float(len(percepts))


def _percept_novelty(percepts: Any) -> float:
    if not isinstance(percepts, list) or not percepts:
        return 0.0
    tail = percepts[-8:]
    seen = {_hash_unit(str(item)) for item in tail}
    return len(seen) / float(len(tail))


def _read_P(state: Any, now: float) -> np.ndarray:
    # Through the shared reading, because half the producers write no
    # timestamp and none of them write `source`. Read raw, the recency of what
    # she saw a moment ago was zero and the count of distinct sources was one
    # for every percept in the stream.
    percepts = _dig(state, "world.recent_percepts", []) or []
    if not isinstance(percepts, list):
        percepts = []
    tail = [read_percept(item, now=now) for item in percepts[-16:]]
    # The strongest claim on attention in the stream, which is the number the
    # workspace prices its perception bid from. This column was the recency of
    # the newest percept, in seconds of wall clock — so two arms of the same
    # trial, run a second apart, differed here by half a standard deviation
    # before anything had happened to either of them. A clock is not state.
    claim = max((item.salience for item in tail), default=0.0)
    sources = {item.kind for item in tail}
    strength = float(np.mean([item.intensity for item in tail])) if tail else 0.0
    objective = _dig(state, "cognition.current_objective", "") or ""
    return np.array(
        [
            # Saturating at four, not sixteen. A percept lives one turn, so a
            # count scaled against sixteen reads a busy turn and a quiet one as
            # the same near-zero.
            _sat(percepts, 4.0),
            claim,
            _sat(sources, 4.0),
            # How much of what arrived has not yet been felt. Novelty over a
            # window that holds one item is one by construction, and that
            # column was a constant for the whole of every run. This one moves
            # within the turn — full before affect has run, empty after — which
            # is the perceptual dynamics the one-turn lifetime actually creates.
            _unfelt_share(percepts),
            strength,
            1.0 if _dig(state, "world.spatial_context") else 0.0,
            _sat(str(objective), 64.0),
            *_content_buckets(objective),
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
            _f(_dig(state, "soma.exertion")),
            _sat(_f((_dig(state, "soma.effort", {}) or {}).get("recall")), 32.0),
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
            _f(_call(organs.free_energy, "get_action_urgency", 0.0, source="organ:free_energy.get_action_urgency")),
            *_ladder(_call(organs.free_energy, "get_trend", "", source="organ:free_energy.get_trend"), _FREE_ENERGY_TREND_LADDER),
        ]
    )
    return np.array(head, dtype=np.float64)


def _read_G(state: Any, organs: Organs) -> np.ndarray:
    focus = _dig(state, "cognition.attention_focus", "") or ""
    workspace = _call(organs.workspace, "get_status", {}, source="organ:workspace.get_status") or {}
    modifiers = _dig(state, "cognition.modifiers", {}) or {}
    return np.array(
        [
            1.0 if focus else 0.0,
            *_content_buckets(focus),
            _f(_dig(state, "cognition.coherence_score"), 1.0),
            _f(_dig(state, "cognition.fragmentation_score")),
            _sat(_f(_dig(state, "cognition.contradiction_count")), 4.0),
            _f(_dig(state, "cognition.conversation_energy"), 0.5),
            _sat(_f(_dig(state, "cognition.discourse_depth")), 8.0),
            _sat(_dig(state, "cognition.discourse_branches", []) or [], 4.0),
            _sat(_dig(state, "cognition.selfhood_reading", {}) or {}, 4.0),
            _f(workspace.get("ignition_level")),
            1.0 if workspace.get("ignited") else 0.0,
            _sat(_f(workspace.get("pending_candidates")), 4.0),
            _f(workspace.get("last_priority")),
            *_content_buckets(workspace.get("last_winner")),
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
        ]
    )
    affect = _call(organs.substrate, "get_substrate_affect", {}, source="organ:substrate.get_substrate_affect") or {}
    status = _call(organs.substrate, "get_status", {}, source="organ:substrate.get_status") or {}
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
    introspection = _call(organs.self_model, "get_introspection", {}, source="organ:self_model.get_introspection") or {}
    beliefs = getattr(organs.self_model, "beliefs", {}) or {}
    agency = _call(organs.agency, "snapshot", {}, source="organ:agency.snapshot") or {}
    prediction = _call(organs.self_prediction, "get_snapshot", {}, source="organ:self_prediction.get_snapshot") or {}
    current = prediction.get("current_prediction") or {}
    comparator = _call(organs.comparator, "get_status", {}, source="organ:comparator.get_status") or {}
    head.extend(
        [
            _sat(_f(introspection.get("belief_count")), 16.0),
            _sat(_f(introspection.get("version")), 32.0),
            _sat(_f(introspection.get("snapshot_count")), 8.0),
            _sat(_f(introspection.get("pending_update_count")), 4.0),
            *_content_buckets(
                " ".join(
                    f"{k}={_content_of(beliefs[k])}"
                    for k in sorted(beliefs)[:16]
                    if str(k).lower() not in _BOOKKEEPING_KEYS
                )
            ),
            _sat(_f(agency.get("acted")), 16.0),
            _f(agency.get("efficacy")),
            _f(agency.get("authored_share")),
            _sat(_f(agency.get("capabilities")), 8.0),
            *_one_hot(agency.get("last_actor", ""), _ACTORS),
            _f(prediction.get("smoothed_error")),
            _sat(_f(prediction.get("surprise_count")), 16.0),
            _f(prediction.get("valence_error_ema")),
            _f(prediction.get("drive_error_ema")),
            _f(prediction.get("focus_error_ema")),
            *_one_hot(prediction.get("most_unpredictable", ""), _UNPREDICTABLE_DIMENSIONS),
            _f(current.get("confidence")),
            _f(comparator.get("agency_score"), 0.5),
            _sat(_f(comparator.get("total_traces")), 16.0),
            _sat(_f(comparator.get("pending_efferences")), 4.0),
            *_ladder(comparator.get("recent_attribution", ""), _ATTRIBUTION_LADDER),
        ]
    )
    return np.array(head, dtype=np.float64)


def _recency(entry: Any, scale: float = 2.0) -> float:
    """How recently this was written, in [0, 1). One is now, zero is long ago.

    The column was the hash of the entry's whole repr, which includes the wall
    clock it was written at — so two arms of one trial, appending the same
    sentence a second apart, differed by the full width of the hash. A hash
    carries no magnitude at the best of times; over a clock it is noise with a
    name, and it was the largest single term in the floor every edge into
    active memory had to clear.

    The scale is two seconds because that is the order of one turn. At sixty
    the whole of a run's spread fell inside three percent of the column, and a
    column that barely moves turns every wobble into several standard
    deviations of nothing.
    """
    if not isinstance(entry, Mapping):
        return 0.0
    stamp = entry.get("timestamp") or entry.get("at") or entry.get("time")
    try:
        age = time.time() - float(stamp)
    except (TypeError, ValueError):
        return 0.0
    if age < 0.0:
        age = 0.0
    return scale / (scale + age)


def _read_M(state: Any) -> np.ndarray:
    working = _dig(state, "cognition.working_memory", []) or []
    retrieved = _dig(state, "cognition.long_term_memory", []) or []
    last = working[-1] if isinstance(working, list) and working else {}
    return np.array(
        [
            _sat(working, 24.0),
            _recency(last),
            _sat(retrieved, 8.0),
            *_content_buckets(" ".join(_content_of(item) for item in list(retrieved)[-4:])),
            max((_f(item) for item in _dig(state, "cognition.memory_scores", []) or []), default=0.0),
            *_content_buckets(" ".join(_content_of(item) for item in list(working)[-4:])),
            _sat(str(_dig(state, "cognition.rolling_summary", "") or ""), 512.0),
            _sat(_dig(state, "cognition.continuity_ledger", {}) or {}, 8.0),
            1.0 if _dig(state, "cognition.active_thread_id") else 0.0,
            _sat(_dig(state, "cold.long_term_memory", []) or [], 64.0),
            _sat(_dig(state, "cold.evolution_log", []) or [], 32.0),
        ],
        dtype=np.float64,
    )


def _surprise_ratio(current: Any, typical: Any) -> float:
    """Surprise against its own running mean, in [0, 1). Half is unremarkable.

    Scale-free on purpose: what matters about a prediction error is whether it
    is larger than this model's errors usually are, and that reading stays
    sensitive wherever the model's absolute error happens to sit.
    """
    now = max(0.0, _f(current))
    usual = max(0.0, _f(typical))
    total = now + usual
    if total <= 1e-9:
        return 0.0
    return now / total


def _read_W(state: Any, organs: Organs) -> np.ndarray:
    facts = _dig(state, "world.facts", {}) or {}
    status = _call(organs.world_model, "status", {}, source="organ:world_model.status") or {}
    facets = status.get("facets", {}) if isinstance(status, Mapping) else {}
    surprise = _call(organs.world_model, "surprise", None, source="organ:world_model.surprise")
    learned = (facets.get("learned", {}) or {}).get("detail", {}) or {}
    causal = (facets.get("causal", {}) or {}).get("detail", {}) or {}
    return np.array(
        [
            _sat(_dig(state, "world.known_entities", {}) or {}, 8.0),
            _sat(_dig(state, "world.relationship_graph", {}) or {}, 8.0),
            _sat(facts, 16.0),
            _sat(_dig(state, "world.user_preferences", {}) or {}, 8.0),
            # What the facts say, not only which facts there are. Read as a
            # list of keys, this column could not move while the world model
            # rewrote the same fact every turn with different content — the
            # keys are stable and the world is not.
            *_content_buckets(
                " ".join(
                    f"{key} {_content_of(facts[key])}"
                    for key in sorted(str(name) for name in facts)[:32]
                )
            ),
            _sat(_dig(state, "cold.concept_graph", {}) or {}, 32.0),
            *_ladder(_dig(state, "cognition.user_emotional_trend", "neutral"), _USER_TREND_LADDER),
            # How surprising this moment is relative to how surprising things
            # usually are, rather than the raw error squashed. Prediction error
            # is unbounded above and `tanh` is flat past about two and a half,
            # so a model whose ordinary error sits at three read 0.995 on every
            # turn and the column was a constant — which is why displacing the
            # world model moved the world model by two hundredths of a standard
            # deviation while reaching three other domains.
            _surprise_ratio(surprise, learned.get("mean_surprise")),
            # In the schema's order. Nine of these seventeen were one place out
            # from `model_hidden_norm` onward: the values were all real and all
            # attached to the wrong names, so every reading of this domain
            # named the wrong quantity — `causal_nodes` was carrying the
            # model's last surprise, and the hidden norm was carrying a count
            # of available facets.
            _sat(_f(learned.get("hidden_norm")), 8.0),
            math.tanh(_f(learned.get("mean_surprise"))),
            math.tanh(_f(learned.get("last_surprise"))),
            _sat(_f(learned.get("step_count")), 10_000.0),
            _sat(_f(causal.get("nodes")), 16.0),
            _sat(_f(causal.get("edges")), 16.0),
            _sat(_f(causal.get("causal_edges")), 8.0),
            _sat(
                [name for name, entry in facets.items() if entry.get("available")], 3.0
            )
            if isinstance(facets, Mapping)
            else 0.0,
            # `train_steps`, not a private `_observations` attribute that does
            # not exist on the object — a feature of my own reading nothing,
            # which is the defect this file was written to find.
            _sat(_f(learned.get("train_steps")), 5_000.0),
        ],
        dtype=np.float64,
    )


def _urgency_of(items: Any) -> float:
    """The hardest any of these intentions is pressing, or nothing."""
    if not isinstance(items, list):
        return 0.0
    best = 0.0
    for item in items:
        if not isinstance(item, Mapping):
            continue
        stated = item.get("urgency")
        if stated is None:
            continue
        best = max(best, _f(stated))
    return max(0.0, min(1.0, best))


def _read_D(state: Any) -> np.ndarray:
    goals = _dig(state, "cognition.active_goals", []) or []
    budgets = _dig(state, "motivation.budgets", {}) or {}
    initiatives = _dig(state, "cognition.pending_initiatives", []) or []
    head = [
        _sat(goals, 8.0),
        _sat(initiatives, 4.0),
        _urgency_of(initiatives),
        _urgency_of(goals),
        # The newest, not the oldest. `goals[:3]` takes the first three, which
        # stop changing the moment there are three — so this domain's whole
        # content profile was a constant after the third turn of every run.
        *_content_buckets(" ".join(_content_of(goal, 320) for goal in goals[-3:])),
        1.0 if str(_dig(state, "cognition.current_origin", "")).startswith("user") else 0.0,
        *_content_buckets(_dig(state, "cognition.last_action_source", "")),
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
    #: Sources that could not be read while this frame was taken, and why. An
    #: empty mapping means every column is a reading; a source in here means
    #: the columns declaring it are a default, and the run has to say so rather
    #: than let a criterion fail on a subsystem nobody asked.
    misses: dict[str, str] = field(default_factory=dict)

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
    and world model. Anything missing reads as zeros — and says so: `misses`
    names every source that could not be read and why, so a failed reader and a
    genuine zero are two states rather than one number.
    """
    moment = time.time() if now is None else now
    kit = organs or Organs()
    with recording_misses() as misses:
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
        misses=dict(misses),
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
    world = _dig(state, "world", None)
    if world is None or not isinstance(getattr(world, "recent_percepts", None), list):
        return False
    # In the shape a percept actually arrives in. The first version wrote a
    # dict with `source` and `salience` and no `type`, which is not a percept
    # any part of the tree produces: the affect phase keys on the type and
    # dropped it, the workspace read a salience nobody writes, and the
    # displacement of perception was a displacement of a list's length.
    #
    # `novel_stimulus` because that is what a probe is — something arrived that
    # was not predicted — and the strength is the displacement itself, in the
    # same units every other domain is displaced in.
    #
    # `world.spatial_context` is deliberately not written. It is a field with
    # no reader anywhere in the tree, so displacing it would move this domain's
    # own vector and could not move anything else: it would inflate the
    # coupling gain with a number that means nothing.
    emit_percept(
        world,
        "novel_stimulus",
        content=f"probe delta {delta:+.4f}",
        intensity=min(1.0, abs(delta)),
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
    # And her own exertion, which is the half of the body that is hers rather
    # than the machine's. An experiment that holds the host still to keep two
    # arms comparable holds still every host channel, so without this the body
    # has no channel left that an intervention can move.
    hit |= _bump(state, "soma.exertion", delta, 0.0, 1.0)
    return hit


def _perturb_A(state: Any, delta: float, ontogeny: Any) -> bool:
    del ontogeny
    hit = _bump(state, "affect.valence", delta, -1.0, 1.0)
    hit |= _bump(state, "affect.arousal", delta, 0.0, 1.0)
    hit |= _bump(state, "affect.curiosity", delta, 0.0, 1.0)
    # And the feeling itself. Ten of this domain's twenty-two columns are
    # emotion channels, and the workspace prices its affect bid by the
    # strongest of them — so a displacement that moved valence and arousal and
    # left the channels alone was a displacement of affect that affect's own
    # consumers could not see. A writer has to move what the reader reads.
    emotions = _dig(state, "affect.emotions", None)
    if isinstance(emotions, dict):
        for name in _EMOTIONS:
            if name in emotions:
                emotions[name] = min(1.0, max(0.0, _f(emotions[name]) + delta))
        hit = True
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
    """The recurrent estimate, and nothing else in the state.

    This also rewrote `cognition.phenomenal_state`'s valence, arousal and
    latent snapshot. That field is rebuilt once a turn from `affect.*`, the
    energy budget and the coherence score, and nothing in the runtime reads any
    of its numbers — so those writes were erased before anything could have
    used them and would have reached nobody if they had survived. Recurrent
    cognition's own state is the liquid substrate and the closed loop, and
    `perturb_organs` is where it is displaced.
    """
    del ontogeny
    return _bump(state, "phi_estimate", delta, -10.0, 10.0)


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
    hit = False
    working = _dig(state, "cognition.working_memory", None)
    if isinstance(working, list):
        working.append(
            {
                "role": "probe",
                "content": f"subject-core memory probe {delta:+.4f}",
                "timestamp": time.time(),
            }
        )
        hit = True
    # And what is in mind. The retrieved set is what recall put there and what
    # the workspace bids a recollection from; displacing active memory without
    # touching it displaces the conversation buffer and calls it memory.
    cognition = getattr(state, "cognition", None)
    if cognition is not None:
        retrieved = list(getattr(cognition, "long_term_memory", []) or [])
        scores = list(getattr(cognition, "memory_scores", []) or [])
        retrieved.append(f"probe recollection {delta:+.4f}")
        # And how strongly it is in mind. Recall writes a match score beside
        # every recollection and the workspace prices its memory bid from it,
        # so a displacement that added text and no score displaced what was
        # recalled without displacing its claim on attention — which is the
        # half of active memory that anything downstream can act on.
        scores.append(min(1.0, max(0.0, 0.5 + delta)))
        cognition.long_term_memory = retrieved[-8:]
        cognition.memory_scores = scores[-8:]
        hit = True
    return hit


def _perturb_W(state: Any, delta: float, ontogeny: Any) -> bool:
    del ontogeny
    facts = _dig(state, "world.facts", None)
    if not isinstance(facts, dict):
        return False
    facts["subject_core_probe"] = {"delta": delta, "at": time.time()}
    # And the entities she takes to be there. A world model displaced only in a
    # scratch dict is displaced in the one part of the world state that nothing
    # consults; who and what is in the room is read by the workspace, the
    # social layer and the schema alike.
    entities = _dig(state, "world.known_entities", None)
    if isinstance(entities, dict):
        entities["subject_core_probe"] = {
            "kind": "probe",
            "salience": min(1.0, max(0.0, 0.5 + delta)),
            "at": time.time(),
        }
    return True


def _perturb_D(state: Any, delta: float, ontogeny: Any) -> bool:
    del ontogeny
    goals = _dig(state, "cognition.active_goals", None)
    hit = False
    if isinstance(goals, list):
        # In the shape the goal engine writes, `priority` included. A goal with
        # no stated priority is a goal the workspace cannot price, and an
        # intention nothing can attend to is not an intention.
        goals.append(
            {
                "id": "subject_core_probe",
                "goal": f"probe intention {delta:+.4f}",
                "description": f"probe intention {delta:+.4f}",
                "origin": "probe",
                "status": "pending",
                "priority": min(1.0, max(0.0, 0.5 + delta)),
                # And what it is asking to be thought about now. The workspace
                # prices deliberation's bid on `urgency` and drops a bid that
                # states none, so a probe goal carrying only a priority was a
                # displacement of deliberation that attention could not see.
                "urgency": min(1.0, max(0.0, 0.5 + delta)),
            }
        )
        hit = True
    budgets = _dig(state, "motivation.budgets", None)
    if isinstance(budgets, dict):
        for name in _DRIVES:
            entry = budgets.get(name)
            if not isinstance(entry, dict):
                continue
            # In units of the budget's own capacity. A drive level runs 0..100,
            # so adding the raw displacement moved it by fifteen hundredths of
            # one percent — a displacement of deliberation that deliberation
            # could not have noticed, in the same way the body's was a
            # displacement of one percent of a CPU.
            span = _f(entry.get("capacity"), 100.0) or 100.0
            for key in ("current", "level"):
                if key in entry:
                    # Toward capacity, by a share of what is missing, rather
                    # than by the same amount everywhere. What deliberation
                    # decides on is which drive is most depleted, and adding
                    # one number to every drive leaves that comparison exactly
                    # as it was: the intention generator dispatched on the same
                    # name in both arms, so the displaced arm did the same
                    # thing as the sham and deliberation reached nothing
                    # through action. Meeting the most pressing need most is
                    # what a displacement of motivation is.
                    level = _f(entry[key])
                    room = max(0.0, span - level)
                    entry[key] = max(0.0, min(span, level + delta * room))
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


async def perturb_organs(
    organs: Organs, domain: str, delta: float, *, state: Any = None
) -> bool:
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
            # Displace the dimensions anything downstream reads. Frustration
            # and curiosity alone moved indices that the homeostatic blend does
            # not look at, so the perturbation was real, gated, applied — and
            # invisible to every consumer of the substrate.
            reading = organs.substrate.get_substrate_affect() or {}
            await organs.substrate.update(
                delta_frustration=delta,
                delta_curiosity=delta,
                valence=min(1.0, max(-1.0, _f(reading.get("valence")) + delta)),
                arousal=min(1.0, max(0.0, _f(reading.get("arousal"), 0.5) + delta)),
                dominance=min(1.0, max(-1.0, _f(reading.get("dominance")) + delta)),
                source="subject_core_probe",
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

            # The runner-up, raised until it wins. A displacement of attention
            # is a change in what wins the competition among what is actually
            # competing — and the first version submitted a candidate of its
            # own under a probe's name, which nothing downstream can interpret:
            # the action she takes is chosen by the source of what she is
            # attending to, and a source no action table knows falls through to
            # the same action the sham took. So displacing the workspace could
            # not change what she did, and the whole route from attention
            # through action to the world and back to perception was closed to
            # the one domain that should open it.
            # Read off what has already been submitted rather than rebuilding
            # the bids: building them reports the work of building them, and a
            # displacement that costs the body something the sham did not pay
            # would manufacture the very edge it is measuring.
            runner_up = None
            lead = 0.0
            pending = list(getattr(workspace, "_candidates", ()) or ())
            if len(pending) >= 2:
                ranked = sorted(
                    pending, key=lambda bid: bid.effective_priority, reverse=True
                )
                runner_up = ranked[1]
                lead = ranked[0].effective_priority
            if runner_up is not None:
                await workspace.submit(
                    CognitiveCandidate(
                        content=runner_up.content,
                        source=runner_up.source,
                        priority=min(1.0, max(0.0, lead + abs(delta))),
                        content_type=runner_up.content_type,
                        affect_weight=runner_up.affect_weight,
                    )
                )
            else:
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
    elif domain == "N":
        # The reservoir the cognitive cycle steps, when that is not the one the
        # state writer already moved. `start_organism` binds the two together,
        # so in an assembled organism this is the same object and displacing it
        # twice would double the intervention; before the organs are up they
        # are different, and displacing only the readout would move what N
        # reports without moving what novelty is computed from.
        try:
            from core.ontogeny.service import get_ontogeny

            reservoir = getattr(get_ontogeny(), "_state", None)
            if (
                reservoir is not None
                and hasattr(reservoir, "h")
                and reservoir is not organs.ontogeny
            ):
                reservoir.h = np.clip(
                    np.asarray(reservoir.h, dtype=np.float64) + delta, -1.0, 1.0
                )
                hit = True
        except Exception:  # noqa: BLE001 - an absent organ is an absent organ
            hit = False
    elif domain == "W" and organs.world_model is not None:
        # An observation vector, which is what the forward model takes. A dict
        # went in and was padded to zeros, so the displacement of the world
        # model was a call that changed nothing: this domain moved itself by
        # seventeen thousandths of a standard deviation while every other
        # domain moved itself by two to ten, and a domain that cannot be
        # displaced cannot be shown to influence anything.
        #
        # The displacement is applied to the situation the cycle actually
        # reports, so the model is shown a world slightly other than the one it
        # is in — which is what displacing a model of the world means.
        try:
            from core.world_model.observe_cycle import action_of, observation_of

            if state is None:
                raise ValueError("displacing the world model needs the situation")
            observation = observation_of(state) + float(delta)
            organs.world_model.observe(observation, action_of(state), learn=True)
            hit = True
        except Exception:  # noqa: BLE001
            hit = False
    return hit
