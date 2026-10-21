"""How eight of the domains are read from her state and organs.

Lifted whole out of `state`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .state import (
        Organs,
        np,
    )


def _read_I(state: Any, organs: Organs) -> np.ndarray:
    from .state import (
        _SENSE_CHANNELS,
        _dig,
        _f,
        _sat,
        np,
    )

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
            # How much is arriving on each of her senses. A channel that
            # reported nothing is absent from the reading rather than written
            # as zero, and reads zero here, which is the same answer for a
            # column and a different fact for the loop that wrote it.
            *(
                _f((_dig(state, "soma.sensors", {}) or {}).get(channel))
                for channel in _SENSE_CHANNELS
            ),
            # And their mean, over the senses she has rather than the ones that
            # happened to report. Linear, so a channel moving by a tenth moves
            # this by a fiftieth instead of by whatever the saturation left.
            sum(
                abs(_f((_dig(state, "soma.sensors", {}) or {}).get(channel)))
                for channel in _SENSE_CHANNELS
            )
            / float(len(_SENSE_CHANNELS)),
            _f((_dig(state, "soma.fatigue", {}) or {}).get("share")),
        ],
        dtype=np.float64,
    )


def _read_A(state: Any, organs: Organs) -> np.ndarray:
    from .state import (
        _DRIVES,
        _EMOTIONS,
        _FREE_ENERGY_TREND_LADDER,
        _WORTH_CHANNELS,
        _call,
        _dig,
        _f,
        _ladder,
        _miss,
        _sat,
        np,
    )

    emotions = _dig(state, "affect.emotions", {}) or {}
    physiology = _dig(state, "affect.physiology", {}) or {}
    head = [
        _f(_dig(state, "affect.valence")),
        _f(_dig(state, "affect.arousal"), 0.5),
        _f(_dig(state, "affect.curiosity"), 0.5),
        _f(_dig(state, "affect.engagement"), 0.5),
        _f(_dig(state, "affect.social_hunger"), 0.5),
        _f(_dig(state, "affect.ambivalence")),
        _f(_dig(state, "affect.confirmation")),
        _f(_dig(state, "affect.delivery_z")),
        1.0 if _dig(state, "affect.breakthrough") else 0.0,
        _f(_dig(state, "affect.steadiness"), 1.0),
        _f(_dig(state, "affect.lift")),
        _f(_dig(state, "affect.frisson")),
        _f(_dig(state, "affect.turn")),
        _f(_dig(state, "affect.safety")),
        _f(_dig(state, "affect.happiness_fear")),
        _f(_dig(state, "affect.change_fear")),
        _f(_dig(state, "affect.borrowed_feeling")),
        _f(_dig(state, "affect.decline_press")),
        _f(_dig(state, "affect.elsewhere_lift")),
        _f(_dig(state, "affect.markers.ambivalence.opposition")),
        _f(_dig(state, "affect.markers.ambivalence.pressure")),
        # A contradiction she is built with, against one she is passing
        # through. Constitutive reads one; anything else, including not yet
        # knowing, reads zero — an unknown standing is not a way.
        1.0 if str(_dig(state, "affect.ambivalence_standing", "") or "") == "the way" else 0.0,
        *(
            1.0 if name in set(_dig(state, "affect.ambivalent_about", ()) or ()) else 0.0
            for name in _DRIVES
        ),
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
    # What she has come to expect of each feeling. The workspace prices an
    # arriving feeling at `intensity - baseline`, so this decides what wins
    # attention, and it was persistent state no column read.
    baselines = _dig(state, "affect.mood_baselines", {}) or {}
    values = [_f(v) for v in baselines.values()] if isinstance(baselines, dict) else []
    head.append(float(np.mean(values)) if values else 0.0)
    head.append(float(np.std(values)) if len(values) > 1 else 0.0)
    moment = _dig(state, "cognition.moment", {}) or {}
    head.extend(_f(moment.get(key)) for key in ("particular", "meets", "weight"))
    homeostasis = _call(
        organs.homeostasis,
        "get_snapshot",
        {},
        source="organ:homeostasis.prospective_dread",
    ) or {}
    head.append(_f(homeostasis.get("prospective_dread")))
    for name in ("_valence_centre", "_valence_scatter", "_arousal_centre", "_arousal_scatter"):
        head.append(_held_number(organs.homeostasis, name, source=f"organ:homeostasis.{name}"))
    head.append(_held_number(organs.wants, "_noise", source="organ:wants._noise"))
    reading = _call(organs.worth, "read", None, source="organ:worth.read")
    error = getattr(reading, "error", {}) or {}
    head.append(_f(getattr(reading, "worth", 0.0)))
    head.append(_f(getattr(reading, "size", 0.0)))
    head.extend(_f(error.get(name)) for name in _WORTH_CHANNELS)
    expected = getattr(organs.worth, "_expected", None)
    if organs.worth is not None and not isinstance(expected, Mapping):
        _miss("organ:worth._expected", "no such reading")
    expected = expected if isinstance(expected, Mapping) else {}
    head.extend(_f(expected.get(name)) for name in _WORTH_CHANNELS)
    news = _call(organs.good_news, "read", None, source="organ:good_news.read")
    head.append(_f(getattr(news, "error", 0.0)))
    head.append(_f(getattr(news, "jump", 0.0)))
    return np.array(head, dtype=np.float64)


def _read_G(state: Any, organs: Organs) -> np.ndarray:
    from .state import (
        _call,
        _content_buckets,
        _dig,
        _earned,
        _f,
        _sat,
        _selfhood,
        np,
    )

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
            *_selfhood(_dig(state, "cognition.selfhood_reading", {}) or {}),
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
            *_earned(organs.credit, workspace.get("last_winner")),
            *_conjunction(state, "attention"),
        ],
        dtype=np.float64,
    )


def _earned(credit: Any, winner: Any) -> list[float]:
    """What the winner has earned, the best and worst earned, and how many sources have."""
    from .state import (
        _call,
        _f,
        _sat,
    )

    held = _call(credit, "read", {}, source="organ:credit.read") or {}
    sources = held.get("sources", {}) if isinstance(held, Mapping) else {}
    values = [_f(entry.get("earned")) for entry in sources.values() if isinstance(entry, Mapping)]
    mine = 0.0
    if credit is not None and winner:
        try:
            mine = float(credit.earned(str(winner)))
        # not a failure: a winner the ledger cannot price has earned nothing yet.
        except (AttributeError, TypeError, ValueError):
            mine = 0.0
    return [
        mine,
        max(values) if values else 0.0,
        min(values) if values else 0.0,
        _sat(float(len(values)), 16.0),
    ]


def _latent(state: Any, width: int) -> list[float]:
    from .state import (
        _dig,
        _f,
        np,
    )

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
    from .state import (
        _FIELD_SOURCES,
        _MODES,
        _call,
        _dig,
        _f,
        _miss,
        _sat,
        _shared_out,
        _sketched,
        np,
    )

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
    # A reading from a snapshot older than the substrate's own freshness bound
    # is not a reading of now. The substrate publishes how old its snapshot is
    # and returns its safe defaults when it cannot answer — plausible numbers
    # that are indistinguishable from a settled state, so nine of recurrent
    # cognition's columns would read as a calm organism whenever the dynamics
    # had stopped. Recorded as a miss, which is what the battery invalidates a
    # criterion on.
    if float(affect.get("snapshot_stale", 0.0) or 0.0) >= 1.0:
        _miss(
            "organ:substrate.get_substrate_affect",
            f"snapshot {float(affect.get('snapshot_age_s', 0.0)):.3g}s old",
        )
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
    head.extend(_sketched(organs.substrate, "x", source="organ:substrate.x"))
    head.extend(_sketched(organs.mesh, "column_activations", source="organ:mesh.column_activations"))
    head.extend(_in_flight(organs.mesh, "_sensory_buffer", source="organ:mesh._sensory_buffer"))
    head.extend(_in_flight(organs.mesh, "_association_buffer", source="organ:mesh._association_buffer"))
    head.extend(_sketched(organs.field, "F", source="organ:field.F"))
    head.extend(_sketched(organs.field, "W_field", source="organ:field.W_field"))
    shares = _call(organs.field, "input_shares", {}, source="organ:field.input_shares") or {}
    head.extend(_f(shares.get(name)) for name in _FIELD_SOURCES)
    head.extend(_sketched(organs.substrate, "W", source="organ:substrate.W"))
    head.extend(_sketched(organs.mesh, "_W_batch", source="organ:mesh._W_batch"))
    head.extend(
        _shared_out(organs.phi_core, "_state_visits", source="organ:phi_core._state_visits")
    )
    head.extend(
        _shared_out(
            organs.phi_core,
            "_affective_state_visits",
            source="organ:phi_core._affective_state_visits",
        )
    )
    head.extend(_sketched(organs.field, "W_chem", source="organ:field.W_chem"))
    head.extend(_conjunction(state, "recurrent"))
    return np.array(head, dtype=np.float64)


def _selfhood(reading: Any) -> list[float]:
    """How much of herself the selfhood tick could read, and what it read.

    Two numbers: the share of the drives that produced a reading, and the mean
    of those readings. A tick that read nothing reads as nothing read and a
    level of zero, which is a different state from a tick that read everything
    and found zero — the first has a share of zero and the second a share of
    one.
    """
    if not isinstance(reading, Mapping):
        return [0.0, 0.0]
    readings = reading.get("readings")
    values = [
        float(one)
        for one in (readings.values() if isinstance(readings, Mapping) else [])
        if isinstance(one, (int, float)) and not isinstance(one, bool)
    ]
    missing = reading.get("missing")
    absent = len(missing) if isinstance(missing, (list, tuple, set)) else 0
    asked = len(values) + absent
    share = (len(values) / asked) if asked else 0.0
    level = (sum(values) / len(values)) if values else 0.0
    return [share, level]


def _shared_out(organ: Any, attribute: str, *, source: str) -> list[float]:
    """An organ's counts as the shape of them, through the shared sketch.

    A count of visits only ever goes one way, so its mean, its smallest and its
    largest are three clocks, and `without_clocks` holds a clock flat because
    elapsed time is not a hidden state. What is state here is which of the
    states she visits often and which rarely, which is the counts divided by
    their total. The share's mean is then one over their number — a constant,
    which `_components` drops — and the rest of the sketch carries the shape.
    """
    from .state import (
        SKETCH_FIELDS,
        _miss,
        np,
        sketch,
    )

    if organ is None:
        _miss(source, "organ absent")
        return [0.0] * len(SKETCH_FIELDS)
    held = getattr(organ, attribute, None)
    if held is None:
        _miss(source, "no such reading")
        return [0.0] * len(SKETCH_FIELDS)
    try:
        counts = np.asarray(held, dtype=np.float64).reshape(-1)
        total = float(np.abs(counts).sum())
    except (TypeError, ValueError):
        _miss(source, "not a count")
        return [0.0] * len(SKETCH_FIELDS)
    if not math.isfinite(total) or total <= 0.0:
        _miss(source, "nothing counted yet")
        return [0.0] * len(SKETCH_FIELDS)
    summary = sketch(counts / total)
    if summary is None:
        _miss(source, "no finite state")
        return [0.0] * len(SKETCH_FIELDS)
    return [summary[part] for part in SKETCH_FIELDS]


def _held_number(organ: Any, attribute: str, *, source: str) -> float:
    """One number an organ keeps as an attribute, a miss when the organ is absent."""
    from .state import _f, _miss

    if organ is None:
        _miss(source, "organ absent")
        return 0.0
    if not hasattr(organ, attribute):
        _miss(source, "no such reading")
        return 0.0
    return _f(getattr(organ, attribute))


def _conjunction(state: Any, meeting: str) -> list[float]:
    """The units of one conjunctive code, at rest when none has been bound.

    Before the first cycle closes, and with the codes switched off, there are
    no units, which is a state of rest rather than a failed reading.
    See core/consciousness/conjunctive_codes.py.
    """
    from .state import CONJUNCTION_UNITS, _f

    codes = getattr(state, "conjunctions", None)
    units = getattr(codes, "units", None)
    values = units.get(meeting) if isinstance(units, Mapping) else None
    if not values:
        return [0.0] * CONJUNCTION_UNITS
    out = [_f(value) for value in list(values)[:CONJUNCTION_UNITS]]
    return out + [0.0] * (CONJUNCTION_UNITS - len(out))


def _in_flight(organ: Any, attribute: str, *, source: str) -> list[float]:
    """An organ's pending input through the shared sketch, and zeros when none is pending.

    The mesh consumes each injection buffer at its next step and sets it back to
    None, so most frames find nothing in flight. That is a reading, no input
    arriving, and it sketches as the zero array would; only an absent organ is
    a miss.
    """
    from .state import SKETCH_FIELDS, _miss, sketch

    if organ is None:
        _miss(source, "organ absent")
        return [0.0] * len(SKETCH_FIELDS)
    held = getattr(organ, attribute, None)
    if held is None:
        return [0.0] * len(SKETCH_FIELDS)
    summary = sketch(held)
    if summary is None:
        return [0.0] * len(SKETCH_FIELDS)
    return [summary[part] for part in SKETCH_FIELDS]


def _sketched(organ: Any, attribute: str, *, source: str) -> list[float]:
    """An organ's state array through the shared sketch, zeros and a miss when absent.

    ``attribute`` may be dotted. The predictive self-model's weights live on the
    executive engine's own model rather than on the engine, and the closure test
    reads them at that depth.
    """
    from .state import (
        SKETCH_FIELDS,
        _miss,
        sketch,
    )

    if organ is None:
        _miss(source, "organ absent")
        return [0.0] * len(SKETCH_FIELDS)
    held: Any = organ
    for step in attribute.split("."):
        held = getattr(held, step, None)
        if held is None:
            break
    summary = sketch(held)
    if summary is None:
        _miss(source, "no finite state")
        return [0.0] * len(SKETCH_FIELDS)
    return [summary[part] for part in SKETCH_FIELDS]


def _read_S(state: Any, organs: Organs) -> np.ndarray:
    from .state import (
        _ACTORS,
        _ATTRIBUTION_LADDER,
        _BOOKKEEPING_KEYS,
        _UNPREDICTABLE_DIMENSIONS,
        _VALUES,
        _call,
        _content_buckets,
        _content_of,
        _dig,
        _f,
        _ladder,
        _one_hot,
        _sat,
        _sketched,
        np,
    )

    growth = _dig(state, "identity.personality_growth", {}) or {}
    # As a mapping: nobody having read her yet is an answer rather than a
    # failed read, and digging field by field would count a miss every turn.
    read_by_other = _dig(state, "identity.read_by_other", {}) or {}
    if not isinstance(read_by_other, Mapping):
        read_by_other = {}
    head = [
        _f(_dig(state, "identity.stability"), 1.0),
        _f(_dig(state, "identity.evolution_score")),
        _f(_dig(state, "identity.bonding_level")),
        _sat(_f(_dig(state, "identity.narrative_version")), 8.0),
        _sat(str(_dig(state, "identity.current_narrative", "") or ""), 512.0),
        _sat(_dig(state, "identity.core_values", []) or [], 8.0),
        _sat(_dig(state, "identity.self_preferences", {}) or {}, 8.0),
        1.0 if read_by_other.get("borrowed") else 0.0,
        _f(read_by_other.get("her_error")),
        _f(read_by_other.get("cared_for")),
        _f((_dig(state, "identity.standing", {}) or {}).get("assigned_share")),
        _f((_dig(state, "identity.standing", {}) or {}).get("tracks_use")),
        _f((_dig(state, "identity.owning_first", {}) or {}).get("lift")),
        _f((_dig(state, "identity.capacity", {}) or {}).get("capacity"), 0.5),
        _f((_dig(state, "identity.met_as_a_type", {}) or {}).get("share")),
        _f((_dig(state, "identity.unchecked_relief", {}) or {}).get("relief")),
        _f((_dig(state, "identity.unchecked_relief", {}) or {}).get("ground")),
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
            _f(prediction.get("confirmation")),
            _sat(_f(prediction.get("confirmation_count")), 16.0),
            _f(prediction.get("confirmation_rate")),
            _f(prediction.get("surprise_rate")),
            _f(prediction.get("conviction")),
            _f(prediction.get("understanding")),
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
    # Defaults of one, not zero: an unbound moment is not a moment she has
    # disowned, and the monitor's own rest value for all three is full
    # ownership. A zero here would read as a self that had lost the world.
    head.extend(
        [
            _f(_dig(state, "cognition.unity_state.agency_ownership_score"), 1.0),
            _f(_dig(state, "cognition.unity_state.self_world_boundary_score"), 1.0),
        ]
    )
    head.extend(
        _sketched(
            organs.executive,
            "_predictive_self.weights",
            source="organ:executive._predictive_self.weights",
        )
    )
    portrait = _call(organs.portrait, "columns", {}, source="organ:portrait.columns") or {}
    head.extend(_f(portrait.get(f"held_{name}")) for name in _VALUES)
    head.extend(_f(portrait.get(f"enacted_{name}")) for name in _VALUES)
    head.append(_f(portrait.get("narrowness")))
    head.append(_f(portrait.get("values_over_drive")))
    head.append(_f(portrait.get("values_foretell_choice")))
    head.extend(_sketched(organs.self_prediction, "_gram", source="organ:self_prediction._gram"))
    return np.array(head, dtype=np.float64)


def _read_M(state: Any) -> np.ndarray:
    from .state import (
        _content_buckets,
        _content_of,
        _dig,
        _f,
        _sat,
        np,
    )

    working = _dig(state, "cognition.working_memory", []) or []
    # As a mapping: no recall yet is an answer rather than a failed read.
    relived = _dig(state, "cognition.relived", {}) or {}
    if not isinstance(relived, Mapping):
        relived = {}
    retrieved = _dig(state, "cognition.long_term_memory", []) or []
    last = working[-1] if isinstance(working, list) and working else {}
    return np.array(
        [
            _sat(working, 24.0),
            # Whether the newest thing in working memory is the user's. This was
            # the newest item's age on the clock, so a memory held still by a
            # lesion aged a frame at a time and moved by 0.47 inside its own
            # clamp. A clock is not state; whose word came last is.
            1.0 if isinstance(last, Mapping) and str(last.get("role", "")).lower() == "user" else 0.0,
            _sat(retrieved, 8.0),
            *_content_buckets(" ".join(_content_of(item) for item in list(retrieved)[-4:])),
            max((_f(item) for item in _dig(state, "cognition.memory_scores", []) or []), default=0.0),
            1.0 if relived.get("relived") else 0.0,
            _f(relived.get("intensity")),
            _sat(_f(relived.get("returns")), 8.0),
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
    from .state import (
        _f,
    )

    now = max(0.0, _f(current))
    usual = max(0.0, _f(typical))
    total = now + usual
    if total <= 1e-9:
        return 0.0
    return now / total


def _read_W(state: Any, organs: Organs) -> np.ndarray:
    from .state import (
        _USER_TREND_LADDER,
        _call,
        _content_buckets,
        _content_of,
        _dig,
        _f,
        _ladder,
        _sat,
        _surprise_ratio,
        np,
    )

    facts = _dig(state, "world.facts", {}) or {}
    # The partner's register is empty until a message has been read, and an
    # empty register is a reading rather than a failed one, so it is taken as
    # a mapping instead of dug into field by field.
    _partner = _dig(state, "world.partner_register", {}) or {}
    if not isinstance(_partner, Mapping):
        _partner = {}
    _pulse = _dig(state, "cognition.partner_cadence", {}) or {}
    if not isinstance(_pulse, Mapping):
        _pulse = {}
    _we = _dig(state, "cognition.togetherness", {}) or {}
    if not isinstance(_we, Mapping):
        _we = {}
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
            _f(_we.get("together")),
            _f(_we.get("edge")),
            _f((_dig(state, "cognition.made_minor", {}) or {}).get("minor")),
            1.0 if str(_dig(state, "cognition.current_partner", "") or "").strip() else 0.0,
            _sat(_f(_pulse.get("chars")), 400.0),
            _sat(_f(_pulse.get("gap")), 60.0),
            _f(_pulse.get("placement")),
            _f(_partner.get("asking")),
            _f(_partner.get("first")),
            _f(_partner.get("second")),
            _f(_partner.get("plural")),
            _sat(_f(_partner.get("persistence")), 10.0),
            1.0 if _partner.get("asks_to_be_witnessed") else 0.0,
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
            # Saturating over a handful of turns: the difference between one
            # turn of silence and three is the whole of what this says, and
            # between thirty and forty it says nothing new.
            _sat(_f(_dig(state, "cognition.turns_since_user_spoke")), 8.0),
            *_conjunction(state, "world"),
        ],
        dtype=np.float64,
    )


def _urgency_of(items: Any) -> float:
    """The hardest any of these intentions is pressing, or nothing.

    `decided_urgency` first. What a proposer declares is a constant chosen at
    its branch, and what the arbiter decides is that proposal shifted by how
    hard the state is actually pressing — so the declared value is a proposal
    and the decided one is the quantity this column is named for. Reading the
    proposal left deliberation with one measured cause in the whole system.
    See core/agency/initiative_arbiter.py `_read_pressure_shift`.
    """
    from .state import (
        _f,
    )

    if not isinstance(items, list):
        return 0.0
    best = 0.0
    for item in items:
        if not isinstance(item, Mapping):
            continue
        stated = item.get("decided_urgency")
        if stated is None:
            stated = item.get("urgency")
        if stated is None:
            continue
        best = max(best, _f(stated))
    return max(0.0, min(1.0, best))


def _read_D(state: Any, organs: Organs) -> np.ndarray:
    from .state import (
        _DRIVES,
        _call,
        _content_buckets,
        _content_of,
        _dig,
        _f,
        _sat,
        _urgency_of,
        np,
    )

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
        # Read as a mapping rather than dug field by field: an empty stance is
        # a reading, "no stance taken", and digging into it would count a miss
        # on every turn nobody testified.
        1.0 if (_dig(state, "cognition.borrowed_resolve", {}) or {}).get("borrowed") else 0.0,
        _f((_dig(state, "cognition.scale", {}) or {}).get("reach")),
        _f((_dig(state, "cognition.scale", {}) or {}).get("pressure")),
        _f((_dig(state, "cognition.persona_gap", {}) or {}).get("gap")),
        1.0 if (_dig(state, "cognition.persona_gap", {}) or {}).get("known_for_the_performance") else 0.0,
        _f((_dig(state, "cognition.constancy", {}) or {}).get("theirs")),
        1.0 if (_dig(state, "cognition.constancy", {}) or {}).get("reallocated") else 0.0,
        _f((_dig(state, "cognition.averted", {}) or {}).get("share_declined")),
        _f((_dig(state, "cognition.averted", {}) or {}).get("held")),
        _f((_dig(state, "cognition.particular", {}) or {}).get("unique")),
        _f((_dig(state, "cognition.particular", {}) or {}).get("anchored")),
        _f((_dig(state, "cognition.closing_window", {}) or {}).get("closing")),
        _f((_dig(state, "cognition.impulse", {}) or {}).get("distrust")),
        _f((_dig(state, "cognition.fuel", {}) or {}).get("share_self")),
        1.0 if (_dig(state, "cognition.fuel", {}) or {}).get("burning_her_own") else 0.0,
        _f((_dig(state, "cognition.returning", {}) or {}).get("pull")),
        1.0 if (_dig(state, "cognition.returning", {}) or {}).get("survives_alternatives") else 0.0,
        _f((_dig(state, "cognition.telling", {}) or {}).get("urge")),
        _sat(_f((_dig(state, "cognition.catharsis", {}) or {}).get("times")), 4.0),
        _f((_dig(state, "cognition.catharsis", {}) or {}).get("drain"), 1.0),
        1.0 if (_dig(state, "cognition.witness", {}) or {}).get("witnessing") else 0.0,
        _f((_dig(state, "cognition.witness", {}) or {}).get("company")),
        _f((_dig(state, "cognition.the_kind", {}) or {}).get("cost_to_source")),
        1.0 if (_dig(state, "cognition.the_kind", {}) or {}).get("mistimed") else 0.0,
        _f((_dig(state, "cognition.never_told", {}) or {}).get("scarcity")),
        _sat(
            _call(organs.intentions, "get_open_intentions", [], source="organ:intentions.open") or [],
            4.0,
        ),
    ]
    for name in _DRIVES:
        entry = budgets.get(name)
        if isinstance(entry, Mapping):
            head.append(_f(entry.get("current", entry.get("level", 0.0))))
        else:
            head.append(_f(entry))
    # And the forces on them this turn. See the schema above for why a budget
    # level cannot carry them.
    forces = _dig(state, "motivation.forces", {}) or {}
    for name in ("pressure", "social_hold", "warmth_return", "attended_credit", "resolve_hold"):
        head.append(_f(forces.get(name)))
    habits = _dig(state, "cognition.habits", {}) or {}
    head.append(_sat(_f(habits.get("to_change")), 4.0))
    head.append(_f(habits.get("largest_deficit")))
    head.append(_f(_call(organs.reserve, "share", 0.0, source="organ:reserve.share")))
    head.extend(_conjunction(state, "deliberation"))
    return np.array(head, dtype=np.float64)


