"""Refactored CognitiveEngine - Now a thin facade over modular phases."""

import asyncio
import json
import logging
import re
import sqlite3
import time
import uuid
from collections import deque
from typing import Any

from core.consciousness.executive_authority import get_executive_authority
from core.conversation.continuation import (
    continuation_prompt_prefix,
    continuation_state_text,
)
from core.goals.objective_lifecycle import (
    finalize_foreground_turn_state,
    is_foreground_objective_origin,
    normalize_objective_origin,
)
from core.memory.retention_policy import working_history_retention_policy
from core.runtime import background_policy, response_policy
from core.runtime.errors import record_degradation
from core.runtime.flags import env_present
from core.runtime.lockdep import LockRank, checked_lock
from core.runtime.pipeline_blueprint import (
    instantiate_legacy_runtime_phases,
    legacy_runtime_phase_specs,
)
from core.runtime.service_registry import get_runtime_service
from core.runtime.structured_input import answer_surface_token_floor
from core.runtime.turn_outcome import (
    TurnOutcome,
    UserVisibleState,
    bind_turn,
    current_turn,
    finalize_turn,
    recoverable_answer,
)
from core.state.aura_state import AuraState, CognitiveMode
from core.utils.concurrency import RobustLock
from core.utils.queues import USER_FACING_ORIGINS
from core.verify import influence_channels
from core.verify.influence_receipt import InfluenceReceipt, build_influence_receipt
from core.verify.lesion_registry import (
    apply_channel,
    get_lesion_registry,
    register_flag_lesion,
)
from core.verify.turn_receipt import (
    TurnReceipt,
    record_phase,
    record_response_path,
    recording_turn,
)

from .autopoiesis import AutopoieticGraph
from .live_mind_contract import (
    REQUIRED_LIVE_MIND_GENERATION_CONTROL_KEYS,
    normalize_live_mind_surface_control_receipt,
)
from .llm.context_assembler import ContextAssembler
from .reasoning_strategies import ReasoningStrategies, StrategyType
from .types import ThinkingMode, Thought

logger = logging.getLogger(__name__)

_USER_FACING_ORIGINS = USER_FACING_ORIGINS

_THOUGHT_HISTORY_LIMIT = working_history_retention_policy(
    "AURA_COGNITIVE_THOUGHT_HISTORY_MAX"
).max_items

_BACKGROUND_REFLECTIVE_MODES = frozenset(
    {
        ThinkingMode.REFLECTIVE,
        ThinkingMode.CREATIVE,
    }
)
_COGNITIVE_ENGINE_RECOVERABLE_ERRORS = (
    AttributeError,
    ConnectionError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)

# A user-facing caller may admit a larger, measured completion surface. The
# former universal 240-second cap silently shortened that admitted deadline;
# on the resident 32B a 2,560-token technical answer then had no possible path
# to completion. Background and unowned cycles retain the original ceiling.
_DEFAULT_COGNITIVE_CYCLE_MAX_S = 240.0


class _RuntimeServiceAdapter:
    """Small compatibility layer for legacy phase constructors expecting container.get."""

    @staticmethod
    def get(name: str, default: Any = None) -> Any:
        return get_runtime_service(name, default=default)


_RUNTIME_SERVICE_ADAPTER = _RuntimeServiceAdapter()


def get_container() -> _RuntimeServiceAdapter:
    """Return the runtime-registry-backed service view used by cognitive phases."""

    return _RUNTIME_SERVICE_ADAPTER


class _NullPassInstrumentation:
    """So the phase loop's contract holds even with no instrumentation."""

    @staticmethod
    def should_run(name: str) -> tuple[bool, int, str]:
        return True, 0, ""


def _pass_instrumentation() -> Any:
    """The pass seam, or a no-op.

    Degrades to a no-op and never to a broken turn: an unavailable debugging
    aid must not be able to stop Aura answering.
    """
    try:
        from core.pipeline.pass_manager import get_instrumentation

        return get_instrumentation()
    except Exception:  # noqa: BLE001 — a debug aid may never break a turn
        logger.debug("pass instrumentation unavailable", exc_info=True)
        return _NullPassInstrumentation()


def _begin_pass_run(label: str) -> None:
    """Number this turn's passes from 1, or do nothing if unavailable.

    Wrapped rather than trusting ``_pass_instrumentation`` to be total: a
    debugging aid must never be the reason a turn fails, and the whole point
    of this seam is that it is on the path every answer takes.
    """
    try:
        begin = getattr(_pass_instrumentation(), "begin_run", None)
        if begin is not None:
            begin(label)
    except Exception:  # noqa: BLE001 — a debug aid may never break a turn
        logger.debug("pass run label %s not recorded", label, exc_info=True)


def _record_legacy_pass(
    name: str,
    ordinal: int,
    duration_s: float,
    *,
    skipped: bool,
    reason: str = "",
    error: str = "",
) -> None:
    """Announce one legacy-pipeline phase to the shared pass record.

    Same ledger the kernel tick writes to, so `AURA_PASS_TRACE=1` and
    `flag_report()` describe the pipeline that is actually serving traffic
    rather than the one that mostly is not.
    """
    try:
        from core.pipeline.pass_manager import PassRecord, get_instrumentation

        get_instrumentation().after_pass(
            PassRecord(
                name=f"legacy_pipeline/{name}",
                ordinal=ordinal,
                duration_s=duration_s,
                skipped=skipped,
                reason=reason,
                error=error,
            )
        )
    except Exception:  # noqa: BLE001 — recording a pass may never break a turn
        logger.debug("pass record dropped for %s", name, exc_info=True)


# ── cognitive provenance on the pipeline that serves chat ────────────────────
#
# Four thin wrappers rather than four inline try/excepts in the phase loop.
# Each one is allowed to fail and none of them may break a turn: a causal
# record that can take the runtime down with it is worse than no record.


def _open_provenance_tick(*, objective: str, priority: bool) -> Any:
    try:
        from core.runtime.cognitive_provenance import open_tick

        return open_tick(objective=str(objective or ""), priority=bool(priority))
    except Exception:  # noqa: BLE001 — provenance may never break a turn
        logger.debug("provenance tick not opened", exc_info=True)
        return None


def _begin_provenance(phase_name: str, state: Any) -> Any:
    try:
        from core.runtime.cognitive_provenance import begin_transformation

        return begin_transformation(phase_name, state)
    except Exception:  # noqa: BLE001
        logger.debug("provenance not started for %s", phase_name, exc_info=True)
        return None


def _complete_provenance(
    transformation: Any, state: Any, *, error: str = "", objective: str = ""
) -> None:
    if transformation is None:
        return
    try:
        transformation.complete(
            state, error=error, inputs={"objective": str(objective or "")[:120]}
        )
    except Exception:  # noqa: BLE001
        logger.debug("provenance receipt dropped", exc_info=True)


def _skip_provenance(phase_name: str, state: Any, reason: str) -> None:
    try:
        from core.runtime.cognitive_provenance import begin_transformation

        begin_transformation(phase_name, state).complete(
            state, skipped=True, skip_reason=str(reason or "")
        )
    except Exception:  # noqa: BLE001
        logger.debug("provenance skip not recorded for %s", phase_name, exc_info=True)


def _close_provenance_tick(graph: Any) -> None:
    if graph is None:
        return
    try:
        from core.runtime.cognitive_provenance import close_tick

        close_tick(graph)
    except Exception:  # noqa: BLE001
        logger.debug("provenance tick not closed", exc_info=True)


def _bounded_float(value: Any, default: float = 0.0, *, lower: float = 0.0, upper: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if parsed != parsed:
        return default
    return max(lower, min(upper, parsed))


def _compact_text(value: Any, *, limit: int = 480) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[: max(0, limit)]
    return text[: limit - 3].rstrip() + "..."


def _combine_advisory_token_factors(factors: list[float]) -> float:
    """Combine advisory max_tokens factors without compounding them.

    Multiple advisory frames (spiking inference, imagination, bicameral,
    cognitive situation) each suggest a budget factor. Multiplying them all
    let four mild 0.75-0.85 reductions compound into a ~35% budget
    (768 → ~250 tokens) that cut live user replies off mid-sentence. Only
    the single strongest reduction applies; boosts apply only when nothing
    asks for a reduction.
    """
    if not factors:
        return 1.0
    reductions = [factor for factor in factors if factor < 1.0]
    return min(reductions) if reductions else max(factors)


_REPLY_TERMINATOR_CHARS = ".!?…\"'”’)]}`"


# Shortest trimmed reply still worth serving instead of losing the turn.
# "The answer is 27." is seventeen characters and is the whole point of the
# turn, so this floor only has to exclude a stub like "Hi." or "Sure.".
_MIN_SALVAGEABLE_REPLY_CHARS = 12

# A sentence boundary the model ran together, e.g. "part of.But — and this is".
# Deliberately narrow: a lowercase letter, terminal punctuation, then a capital
# that starts a lowercase word. Digits are excluded so decimals and version
# numbers survive, and a single capital (U.S.A, initials) will not match because
# the following character must be lowercase.
# Three shapes of run-on, because the first version only caught one of them.
#
# Measured live 2026-07-27: "...leading into Q3?Finally, think about how
# critical this rewrite is" and "...certain states get reinforced.I wouldn't
# call it preference". The original required a LOWERCASE letter before the
# terminator and a capital-plus-lowercase after, so a digit before ("Q3?") and
# a lone capital after (".I ") both slipped through.
#
# "?" and "!" never appear inside an identifier, so a digit may precede them
# safely. The "." case stays conservative — lowercase before, and either a
# normal capitalised word or the pronoun "I" after — because "config.Name" has
# the same shape as a run-on and must not be broken.
_RUN_ON_SENTENCE_RE = re.compile(
    r"(?<=[a-z0-9])([?!])([A-Z])"
    r"|(?<=[a-z])([.])([A-Z][a-z])"
    r"|(?<=[a-z])([.])(I\s)"
)


def _rejoin_run_on(match: "re.Match[str]") -> str:
    groups = [group for group in match.groups() if group is not None]
    return f"{groups[0]} {groups[1]}"



# Questions whose honest answer is a derivation, not a sentence. Budget cut
# these off mid-working, and a derivation without its conclusion is not a
# partial answer — it is no answer, delivered confidently.
_DERIVATION_CUE_RE = re.compile(
    r"\b(?:step[- ]by[- ]step|show your work(?:ing)?|derive|derivation|prove|"
    r"work (?:it|this) out|walk me through|how did you get|explain how|"
    r"calculate|compute|how (?:far|long|many|much)|when does|what time)\b",
    re.IGNORECASE,
)
# Two questions asked at once, or one question with a second quantity in it.
# Deliberately narrow: "what would it be and why that one?" is conversation,
# not a derivation, and treating it as one lengthens ordinary turns for nothing.
_MULTI_PART_QUESTION_RE = re.compile(
    r"\?[^?]*\?"
    r"|\b(?:and|then)\b[^.?!]{0,60}\b(?:how (?:far|long|many|much|fast|old)|"
    r"what (?:time|number|value|percentage|fraction)|how do (?:i|you|we) get)\b",
    re.IGNORECASE,
)


def _turn_wants_a_derivation(user_message: str) -> bool:
    """Does answering this honestly take working, or just a sentence?"""
    text = " ".join(str(user_message or "").split())
    if not text or len(text) > 1200:
        return False
    if _DERIVATION_CUE_RE.search(text):
        return True
    return bool(_MULTI_PART_QUESTION_RE.search(text))


def _restore_sentence_spacing(text: str) -> str:
    """Put back the space between sentences the surface ran together.

    Measured live 2026-07-26, in an otherwise excellent 964-character reply:
    "…losing parts of myself or the world I've been part of.But — and this is
    where it gets complicated…", "…because they're redundant.The mercy part…",
    "…how I understand this world.What about you?" — every paragraph boundary
    arrived with its whitespace gone.

    Nothing in the serving path removes newlines, so this is the model emitting
    them that way. It is still what the person reads, and one space is a safe
    repair: code fences are left alone, and the pattern cannot fire on decimals,
    initials, or abbreviations.
    """
    body = str(text or "")
    # Any backtick means code is present — a fence, or inline `obj.Method`.
    # `file.Name` matches the same shape as a run-on sentence, and inserting a
    # space there would corrupt an identifier, so prose-only is the safe scope.
    if not body or "`" in body:
        return body
    return _RUN_ON_SENTENCE_RE.sub(_rejoin_run_on, body)


def _trim_midsentence_cutoff(text: str) -> tuple[str, bool]:
    """Backstop for replies that stop mid-clause at the token budget.

    A user-facing turn must never end on a dangling fragment like
    "Weighted against" — if the tail is clearly unfinished and a sentence
    boundary exists in the final 40% of the text, cut there. Returns the
    (possibly trimmed) text and whether a trim happened. Keeps the text
    untouched when no safe boundary exists: a partial answer still beats
    an empty one.
    """
    stripped = str(text or "").rstrip()
    if not stripped:
        return stripped, False
    if stripped[-1] in _REPLY_TERMINATOR_CHARS or stripped.endswith("```"):
        return stripped, False
    last_boundary = max(stripped.rfind(ch) for ch in ".!?…")
    # Keep whatever complete sentences exist, measured in what SURVIVES rather
    # than as a fraction of what was generated. The old rule required the
    # boundary to fall in the last 40% of the text, so a reply that answered
    # early and then ran into its token budget kept the dangling clause, failed
    # the reliability gate as `truncated_tail`, and was discarded whole.
    #
    # Live 2026-07-26: "Cortex response received (len=366)" then
    # "reply_reliability_gate_failed:truncated_tail" then "Skipping
    # CognitiveEngine desktop repair retry" — and the person was handed "I
    # couldn't get to an answer I'd stand behind" in place of the 366
    # characters she had actually produced. Cutting to "The answer is 27." is
    # worth far more than losing the turn, even when most of the draft goes
    # with the unfinished clause.
    salvaged = stripped[: last_boundary + 1] if last_boundary >= 0 else ""
    if len(salvaged) >= _MIN_SALVAGEABLE_REPLY_CHARS:
        return salvaged, True
    return stripped, False


def _truncation_verdict(text: str, *, generation_stop_reason: str = "") -> bool:
    """Ask the gate that will JUDGE this reply whether the tail is truncated."""

    try:
        from core.conversation.response_reliability import _has_truncated_tail
    except (ImportError, AttributeError):
        return False
    try:
        return bool(
            _has_truncated_tail(
                text,
                generation_stop_reason=generation_stop_reason,
            )
        )
    except (RuntimeError, TypeError, ValueError):
        return False


def _complete_reply_tail(text: str) -> tuple[str, bool]:
    """Trim a clipped reply until the reliability gate accepts its tail.

    `_trim_midsentence_cutoff` judges completeness by the LAST CHARACTER, while
    the reliability gate that decides whether the turn lives applies a much
    richer test — unmatched quotes, dangling gerunds, trailing conjunctions,
    orphaned list numbers. Two different rules on the same text means the
    trimmer can declare a reply finished and the gate can still reject it, and
    then the turn dies with a real answer in hand: measured live, "Cortex
    response received (len=240)" followed immediately by
    "reply_reliability_gate_failed:truncated_tail" and the person was handed
    "I couldn't get to an answer I'd stand behind on that one."

    So trim against the detector that grades the result, one sentence boundary
    at a time, and stop as soon as it is satisfied. If nothing survivable
    remains, return the text untouched — a partial answer still beats none, and
    that is the caller's existing behaviour.
    """

    stripped = str(text or "").rstrip()
    if not stripped:
        return stripped, False

    trimmed, did_trim = _trim_midsentence_cutoff(stripped)
    if not _truncation_verdict(trimmed):
        return trimmed, did_trim

    candidate = trimmed
    # Bounded: each pass removes at least one sentence, so a handful of passes
    # either satisfies the gate or proves nothing here will.
    for _ in range(6):
        boundary = max(candidate.rfind(char) for char in ".!?…")
        if boundary < 0:
            break
        candidate = candidate[:boundary].rstrip()
        boundary = max(candidate.rfind(char) for char in ".!?…")
        candidate = candidate[: boundary + 1].rstrip() if boundary >= 0 else ""
        if len(candidate) < _MIN_SALVAGEABLE_REPLY_CHARS:
            break
        if not _truncation_verdict(candidate):
            return candidate, True
    return trimmed, did_trim


def _compact_json(value: Any, *, limit: int = 2400) -> str:
    try:
        text = json.dumps(value, sort_keys=True, default=str, ensure_ascii=True)
    except (TypeError, ValueError):
        text = str(value or "")
    return _compact_text(text, limit=limit)


def _nested_value(data: Any, path: tuple[str, ...], default: Any = None) -> Any:
    current = data
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return current if current is not None else default


def _nested_float(
    data: Any,
    path: tuple[str, ...],
    default: float = 0.0,
    *,
    lower: float = 0.0,
    upper: float = 1.0,
) -> float:
    return _bounded_float(_nested_value(data, path, default), default, lower=lower, upper=upper)


def _turn_needs_undistorted_computation(user_message: Any) -> bool:
    """Whether this turn has one right answer that affect must not bend.

    Substrate steering exists to give her replies her own voice. It works by
    pushing an affect direction into the residual stream, and on a turn whose
    answer is a fact rather than a feeling that push is pure distortion.

    Measured live on the desktop surface 2026-07-26. "What is 17 minus 8, and
    then times 3?" — a question the bare 32B answers without effort — came back
    twice, from a healthy resident cortex:

        "Not too broad. Some skills serve me better than others.Did you pay
         attention in class? Hey, look at this - ätze! I got chocolate on my
         shirt."

        "Five thousand: So first you break it down. Mental operations can
         generate digits well outside the world's population. Imagination
         defaults to scalar scaling when pushed into math without boundaries."

    The second is the tell: not noise, but the answer being pulled toward
    whatever the affect vector encodes — talking ABOUT scaling and imagination
    instead of subtracting eight from seventeen.
    """
    try:
        from core.brain.executable_reasoning import (
            ReasoningObjectiveRole,
            classify_reasoning_objective_role,
        )
        from core.conversation.response_reliability import (
            asks_for_a_number,
            requires_reasoning_lane,
        )
    except ImportError:
        return False
    try:
        if (
            classify_reasoning_objective_role(str(user_message or ""))
            is ReasoningObjectiveRole.EXPOSITORY
        ):
            return False
        return bool(
            asks_for_a_number(user_message) or requires_reasoning_lane(user_message)
        )
    except (RuntimeError, TypeError, ValueError):
        return False


def _apply_neurodynamic_sampling_bias(
    controls: dict[str, Any],
    advice: Any,
) -> dict[str, Any]:
    """Let the spiking model change the decode, not just the prompt.

    ``SpikingActiveInferenceAdvisor._sampling_bias`` computes three numbers
    from the neurodynamics: ``temperature_delta``, ``top_p_delta`` and
    ``max_tokens_factor``. Only the last one was ever read. The first two were
    computed on every turn and dropped, so the spiking model's entire route to
    behaviour was ``_compact_spiking_active_inference_directive`` — English
    sentences appended to the prompt, "Neurodynamic advisory: Keep the reply
    compact and stable because runtime load pressure is elevated."

    That is what "advisory-only, decoupled from the decision pipeline" meant in
    practice: a neurodynamic model whose only actuator was a sentence. Asking
    the model nicely is not a mechanism.

    The deltas now move the actual decode parameters, inside the same bounds
    the affective controls already respect — so uncertainty and error pressure
    narrow the distribution the tokens are drawn from, and novelty widens it.
    Bounded, auditable, and falsifiable: change the neurodynamics and the
    sampler changes.
    """
    if not controls or not isinstance(advice, dict):
        return controls
    # Lesioned, the deltas are dropped and the decode runs at whatever the mind
    # controls already said — which is precisely the state this function was
    # written to end, so it is also the counterfactual that measures whether
    # ending it changed anything.
    if get_lesion_registry().is_lesioned(influence_channels.SPIKING_SAMPLING_BIAS):
        return controls
    sampling = advice.get("sampling_bias")
    if not isinstance(sampling, dict):
        return controls

    def _delta(key: str, limit: float) -> float:
        try:
            value = float(sampling.get(key, 0.0))
        except (TypeError, ValueError):
            return 0.0
        if not (-limit <= value <= limit):
            return 0.0
        return value

    temperature_delta = _delta("temperature_delta", 0.25)
    top_p_delta = _delta("top_p_delta", 0.25)
    if not temperature_delta and not top_p_delta:
        return controls

    updated = dict(controls)
    if "temperature" in updated and temperature_delta:
        updated["temperature"] = round(
            max(0.22, min(0.82, float(updated["temperature"]) + temperature_delta)), 4
        )
    if "top_p" in updated and top_p_delta:
        updated["top_p"] = round(
            max(0.72, min(0.94, float(updated["top_p"]) + top_p_delta)), 4
        )
    updated["neurodynamic_sampling_applied"] = {
        "temperature_delta": temperature_delta,
        "top_p_delta": top_p_delta,
    }
    return updated


#: What the control policy below IS. Every weight, threshold and clamp in it
#: was chosen by hand — there is no model-specific calibration behind them, no
#: uncertainty propagated through them, no held-out evidence that these
#: particular numbers beat neighbouring ones, and no learned policy that
#: produced them. The mechanism is real and causal: it moves temperature, top_p
#: and recurrent depth, and the lesion registry can run a turn without it
#: (see _register_live_mind_lesions). What it is not is calibrated, and the
#: receipt says so rather than leaving a reader to assume otherwise.
LIVE_MIND_CONTROL_POLICY = "hand_tuned_heuristic.v1"
LIVE_MIND_CONTROL_POLICY_CALIBRATED = False


def _live_mind_generation_controls(
    live_mind_context: Any,
    *,
    user_message: Any = None,
) -> dict[str, Any]:
    """Map a mind snapshot onto sampling controls.

    A hand-tuned heuristic — see LIVE_MIND_CONTROL_POLICY. Causal and
    measurable through the lesion registry; not calibrated, and not presented
    as such anywhere downstream.
    """
    if not isinstance(live_mind_context, dict):
        return {}
    quality = live_mind_context.get("mind_snapshot_quality")
    if not isinstance(quality, dict) or not bool(quality.get("ready")):
        return {}
    snapshot = live_mind_context.get("mind_snapshot")
    if not isinstance(snapshot, dict):
        return {}

    dominant_label = str(
        _nested_value(snapshot, ("affect_grounding", "dominant", "label"), "")
    ).lower()
    dominant_intensity = _nested_float(
        snapshot, ("affect_grounding", "dominant", "intensity"), 0.0
    )
    curiosity_drive = _nested_float(
        snapshot, ("drive_integration", "drives", "curiosity", "activation"), 0.0
    )
    pain = _nested_float(snapshot, ("nociception", "nociceptive_pressure"), 0.0)
    integration = _nested_float(snapshot, ("phenomenal_engine", "integration"), 0.0)
    self_presence = _nested_float(snapshot, ("phenomenal_engine", "self_presence"), 0.5)
    self_knowing_pressure = _nested_float(
        snapshot,
        ("automatic_self_knowing", "controls", "self_knowing_pressure"),
        0.0,
    )
    second_order_strength = _nested_float(
        snapshot,
        ("recursive_self_knowing", "latest", "second_order_strength"),
        0.0,
    )
    phenomenal_knowing = _nested_float(
        snapshot,
        ("phenomenal_knowing", "controls", "phenomenal_knowing"),
        0.0,
    )
    expectation_error = _nested_float(
        snapshot, ("outcome_ledger", "expectation_calibration"), 0.0
    )
    workspace_ignited = bool(_nested_value(snapshot, ("global_workspace", "ignited"), False))

    curiosity = max(curiosity_drive, dominant_intensity if dominant_label == "curiosity" else 0.0)
    distress = max(
        pain,
        dominant_intensity if dominant_label in {"anxiety", "frustration", "upset"} else 0.0,
        expectation_error,
    )

    temperature = 0.58
    top_p = 0.88
    steering_alpha = 0.0
    recurrent_loops = 1

    if curiosity >= 0.45:
        temperature += min(0.08, curiosity * 0.08)
        top_p += min(0.04, curiosity * 0.04)
    if distress >= 0.25:
        temperature -= min(0.14, distress * 0.18)
        top_p -= min(0.10, distress * 0.14)
        recurrent_loops = 2
    if workspace_ignited or integration >= 0.60:
        top_p -= 0.03
    if self_presence <= 0.35:
        temperature -= 0.05
        recurrent_loops = 2
    if curiosity >= 0.65 and distress < 0.20:
        recurrent_loops = 2
    if self_knowing_pressure >= 0.50 or phenomenal_knowing >= 0.60:
        recurrent_loops = max(recurrent_loops, 2)
    if second_order_strength >= 0.75:
        temperature -= 0.02

    if _turn_needs_undistorted_computation(user_message):
        # Determinate computation also uses one clean forward pass. Residual
        # steering is already neutral for every user-facing decode until it
        # earns model-specific no-regression authority.
        return {
            "temperature": round(min(temperature, 0.30), 4),
            "top_p": round(min(top_p, 0.90), 4),
            "clean_user_surface_recurrent_loops": 1,
            "clean_user_surface_steering_alpha": 0.0,
        }

    temperature, top_p, recurrent_loops = _apply_functional_i_constraint(
        temperature, top_p, recurrent_loops
    )

    return {
        "temperature": round(max(0.22, min(0.82, temperature)), 4),
        "top_p": round(max(0.72, min(0.94, top_p)), 4),
        "clean_user_surface_recurrent_loops": recurrent_loops,
        "clean_user_surface_steering_alpha": round(max(0.0, min(1.0, steering_alpha)), 4),
    }


def _apply_functional_i_constraint(
    temperature: float,
    top_p: float,
    recurrent_loops: int,
) -> tuple[float, float, int]:
    """Let the functional "I" tighten this turn's sampling, never loosen it.

    FunctionalIAttractor and ClosedLoopPolicyCoupler compute continuity,
    coherence, integrity, identity tension, agency readiness and first-person
    confidence, and map them onto temperature, top-p, planning depth,
    verification threshold, retrieval depth and tool risk. Neither class had a
    production call path, so all of that was computed by nothing, for nothing —
    and the attractor's own docstring says it is real only when "policy changes
    downstream". This is that leg.

    **Tighten only.** The coupler's caution term is the direction its outputs
    mean something in: identity tension and trust debt lower temperature and
    raise verification. Letting it RAISE temperature would mean a confident
    self-model buys more randomness, which is not what any of its terms
    measure, and it would make the self-model a licence rather than a brake.
    ``min`` in both directions is the honest reading of what the coupler
    computes.

    An absent policy leaves the turn unchanged. Before the first BeingRuntime
    sample there is no "I" to consult, and treating that silence as calm is
    exactly the "absence of a check reported as a passed check" failure this
    codebase has a standing finding for.
    """

    try:
        from core.being.runtime import get_being_runtime

        policy = get_being_runtime().closed_loop_policy()
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "cognitive_engine",
            exc,
            severity="warning",
            action="generated without functional-I sampling constraint for this turn",
        )
        return temperature, top_p, recurrent_loops
    if policy is None:
        return temperature, top_p, recurrent_loops

    constrained_temperature = min(float(temperature), float(policy.temperature))
    constrained_top_p = min(float(top_p), float(policy.top_p))
    # A raised verification threshold is the coupler saying "check more before
    # you speak". A second recurrent pass is what this lane has to spend on
    # that, so the two are connected here rather than left as a number in a
    # dataclass nobody reads.
    constrained_loops = recurrent_loops
    if float(policy.verification_threshold) >= 0.70:
        constrained_loops = max(recurrent_loops, 2)
    return constrained_temperature, constrained_top_p, constrained_loops


def _live_mind_controls_bound(
    live_mind_context: Any,
    generation_controls: Any,
) -> bool:
    """Whether a live-mind snapshot may steer generation for this turn.

    The snapshot arrives inside the caller's ``context`` dict and used to
    vouch for itself: its own ``ready`` flag said it was ready, and nothing
    established that this runtime had produced it. ``think()`` accepts an
    arbitrary context, so anything reaching that entry point could hand over a
    dictionary and take control of temperature, top_p and recurrent depth.

    Two conditions now: the payload carries this process's stamp, and the
    subsystems the snapshot itself declares as required are actually healthy.
    ``required_subsystems_ok`` was computed and recorded beside this and never
    consulted by it — a check that ran, produced an answer, and gated nothing.
    """
    if not isinstance(live_mind_context, dict) or not isinstance(generation_controls, dict):
        return False
    from core.utils.injected_blocks import is_stamped_runtime_payload

    if not is_stamped_runtime_payload(live_mind_context):
        logger.debug(
            "Live-mind controls refused: the snapshot carries no runtime stamp."
        )
        return False
    quality = live_mind_context.get("mind_snapshot_quality")
    snapshot = live_mind_context.get("mind_snapshot")
    if not isinstance(quality, dict) or not bool(quality.get("ready")):
        return False
    if not isinstance(snapshot, dict):
        return False
    if not bool(live_mind_context.get("required_subsystems_ok")):
        logger.debug(
            "Live-mind controls refused: required subsystems are not healthy."
        )
        return False
    return REQUIRED_LIVE_MIND_GENERATION_CONTROL_KEYS.issubset(
        generation_controls.keys()
    )


#: Steering off. Zero is admitted end-to-end; an inactive optional modifier
#: must not be confused with an unavailable cortex or an unavailable voice.
_STEERING_OFF = 0.0

#: One forward pass: the neutral for recurrent depth.
_SINGLE_PASS = 1


def _register_live_mind_lesions() -> None:
    """Make the live-mind channels neutralizable, and therefore measurable.

    Each of these already reaches generation. What none of them had was a way
    to run the turn without it, which is the only thing that can distinguish a
    channel that shapes the reply from one that is merely computed.
    """

    register_flag_lesion(
        influence_channels.LIVE_MIND_GENERATION_CONTROLS,
        owner="core/brain/cognitive_engine.py",
        neutral="temperature and top_p omitted; the router samples at its own defaults",
        direct_actuation=True,
    )
    register_flag_lesion(
        influence_channels.LIVE_MIND_STEERING_ALPHA,
        owner="core/brain/cognitive_engine.py",
        neutral=f"steering alpha forced to {_STEERING_OFF} (off, inside the admitted range)",
        direct_actuation=True,
    )
    register_flag_lesion(
        influence_channels.LIVE_MIND_RECURRENT_LOOPS,
        owner="core/brain/cognitive_engine.py",
        neutral=f"{_SINGLE_PASS} recurrent pass: the answer is read off a clean forward pass",
        direct_actuation=True,
    )
    register_flag_lesion(
        influence_channels.LIVE_MIND_CONTEXT_BLOCK,
        owner="core/brain/cognitive_engine.py",
        neutral="the [LIVE MIND CONTEXT] block is omitted from the system prompt entirely",
        direct_actuation=False,
    )
    for channel, source in (
        (influence_channels.SPIKING_SAMPLING_BIAS, "spiking active inference"),
        (influence_channels.IMAGINATION_SAMPLING_BIAS, "the imagination workspace"),
        (influence_channels.BICAMERAL_SAMPLING_BIAS, "the bicameral advisory"),
    ):
        register_flag_lesion(
            channel,
            owner="core/brain/cognitive_engine.py",
            neutral=f"no sampling bias from {source} reaches the decode",
            direct_actuation=True,
        )


_register_live_mind_lesions()


def _attach_turn_receipt(thought: Any, receipt: TurnReceipt) -> None:
    """Travel the path evidence with the answer it explains.

    Attached to the Thought rather than logged, because the consumer who needs
    it is whoever is about to describe what this reply demonstrates.
    """

    metadata = getattr(thought, "metadata", None)
    if isinstance(metadata, dict):
        metadata["turn_receipt"] = receipt.as_dict()


def live_mind_influence_receipt(source: str) -> InfluenceReceipt:
    """What is actually measured about the channels this turn claims to use.

    Provenance and causality are different questions and this codebase has been
    answering the first while readers heard the second. A control derived from
    a real snapshot and applied to a real sampler is bound; whether it moved
    the reply is only knowable from paired trials against a measured null, and
    for most channels nobody has run one. This says so rather than implying
    otherwise by omission.
    """

    return build_influence_receipt(
        (
            influence_channels.LIVE_MIND_GENERATION_CONTROLS,
            influence_channels.LIVE_MIND_STEERING_ALPHA,
            influence_channels.LIVE_MIND_RECURRENT_LOOPS,
            influence_channels.LIVE_MIND_CONTEXT_BLOCK,
        ),
        source=source,
    )


def _bind_live_mind_generation_contract(context: dict[str, Any]) -> dict[str, Any]:
    """Bind one authoritative mind-state control contract to a cognitive turn."""

    live_mind_context = context.get("live_mind_context")
    generation_controls = _live_mind_generation_controls(
        live_mind_context,
        user_message=context.get("visible_user_message"),
    )
    if generation_controls:
        from core.brain.llm.user_surface_recurrence import (
            admit_user_surface_recurrent_loops,
        )

        generation_controls["clean_user_surface_recurrent_loops"] = (
            admit_user_surface_recurrent_loops(
                generation_controls.get("clean_user_surface_recurrent_loops")
            )
        )
    controls_bound = _live_mind_controls_bound(
        live_mind_context,
        generation_controls,
    )
    snapshot_ready = bool(
        isinstance(live_mind_context, dict)
        and isinstance(live_mind_context.get("mind_snapshot_quality"), dict)
        and live_mind_context["mind_snapshot_quality"].get("ready")
    )
    required_subsystems_ok = bool(
        isinstance(live_mind_context, dict)
        and live_mind_context.get("required_subsystems_ok")
    )
    desktop_required = bool(
        context.get("desktop_cognitive_engine_required", False)
        or context.get("cognitive_engine_required", False)
    )

    context["live_mind_generation_controls"] = dict(generation_controls)
    context["live_mind_controls_bound"] = controls_bound
    # Travels with the binding, so nothing downstream reads these numbers as
    # a calibrated policy.
    context["live_mind_control_policy"] = {
        "policy": LIVE_MIND_CONTROL_POLICY,
        "calibrated": LIVE_MIND_CONTROL_POLICY_CALIBRATED,
        "evidence": "lesionable_via_influence_channels",
    }
    context["live_mind_snapshot_ready"] = snapshot_ready
    context["live_mind_required_subsystems_ok"] = required_subsystems_ok
    context["clean_user_surface_contract"] = bool(
        context.get("clean_user_surface_contract", False) or desktop_required
    )
    return generation_controls


#: Reported once per shape: a caller that always passes unattested history
#: would otherwise fill the trail.
_UNATTESTED_EXCHANGE_SEEN: set[str] = set()


def _note_unattested_exchange(entry: Any) -> None:
    shape = ",".join(sorted(str(key) for key in entry)) if isinstance(entry, dict) else "non_dict"
    if shape in _UNATTESTED_EXCHANGE_SEEN:
        return
    _UNATTESTED_EXCHANGE_SEEN.add(shape)
    logger.warning(
        "🔏 Dropped a conversation exchange with no runtime stamp (keys: %s). "
        "Its producer should call injected_blocks.stamp_runtime_payload().",
        shape,
    )


def _desktop_history_messages_from_context(
    context: dict[str, Any],
    *,
    max_pairs: int = 4,
) -> list[dict[str, str]]:
    from core.conversation.delivered_history import delivered_exchange_messages

    return delivered_exchange_messages(
        context.get("recent_completed_exchanges"),
        max_pairs=max_pairs,
        on_unattested=_note_unattested_exchange,
    )


def _record_objective_binding(
    state: AuraState, objective: str, *, source: str, mode: Any, reason: str
) -> None:
    try:
        mode_value = getattr(mode, "value", mode)
        get_executive_authority().record_objective_binding(
            state,
            objective,
            source=source,
            mode=str(mode_value or ""),
            reason=reason,
        )
    except (RuntimeError, AttributeError, TypeError) as exc:
        record_degradation(
            "cognitive_engine",
            exc,
            severity="warning",
            action="skipped executive objective audit and continued cognition",
        )
        logger.debug("Executive objective audit skipped for %s: %s", source, exc)


def _compact_spiking_active_inference_directive(advice: dict[str, Any] | None) -> str:
    if not isinstance(advice, dict):
        return ""
    action = str(advice.get("action") or "").strip()
    routing = advice.get("routing_bias") or {}
    if not isinstance(routing, dict):
        routing = {}
    working_memory = advice.get("working_memory") or {}
    if not isinstance(working_memory, dict):
        working_memory = {}
    uncertainty = advice.get("uncertainty", 0.0)
    try:
        uncertainty_value = float(uncertainty)
    except (TypeError, ValueError):
        uncertainty_value = 0.0

    directives: list[str] = []
    if bool(routing.get("ask_clarification")):
        directives.append("If the request is underspecified, ask one precise clarifying question.")
    if bool(routing.get("seek_information")):
        directives.append("If current facts matter, explain what should be verified before acting.")
    if bool(routing.get("use_tool_gateway")):
        directives.append("For external effects, describe the governed tool path and do not claim tool completion without evidence.")
    if bool(routing.get("reduce_load")):
        directives.append("Keep the reply compact and stable because runtime load pressure is elevated.")
    if working_memory.get("admission") == "compress_foreground":
        directives.append("Preserve the user intent while compressing nonessential detail under working-memory pressure.")
    if bool(routing.get("repair_first")):
        directives.append("Prioritize diagnosis and repair steps before speculative explanation.")
    if not directives and action:
        directives.append(f"Current advisory tendency: {action.replace('_', ' ')}.")
    if uncertainty_value >= 0.65:
        directives.append("State uncertainty plainly rather than guessing.")

    if not directives:
        return ""
    return "Neurodynamic advisory: " + " ".join(directives)


def _compact_imagination_directive(frame: dict[str, Any] | None) -> str:
    if not isinstance(frame, dict):
        return ""
    try:
        salience = float(frame.get("salience", 0.0) or 0.0)
    except (TypeError, ValueError):
        salience = 0.0
    if salience < 0.18:
        return ""

    routing = frame.get("routing_bias") or {}
    if not isinstance(routing, dict):
        routing = {}
    directives = [
        "Imagination workspace: use the internal hypothetical model to enrich the answer, but do not claim it is observed reality."
    ]
    visual = str(frame.get("visual_model") or "").strip()
    bridge = str(frame.get("conceptual_bridge") or "").strip()
    phrase = str(frame.get("phrase_model") or "").strip()
    canvas = frame.get("mental_canvas") or {}
    if not isinstance(canvas, dict):
        canvas = {}
    image_prompt = str(canvas.get("image_prompt") or "").strip()
    novel_thoughts = frame.get("novel_thoughts") or []
    if visual:
        directives.append(f"Imagined visual model: {visual[:220]}")
    if image_prompt:
        directives.append(f"Mental canvas: {image_prompt[:220]}")
    if bridge:
        directives.append(f"Novel connection: {bridge[:220]}")
    if phrase:
        directives.append(f"Linguistic seed: {phrase[:160]}")
    if isinstance(novel_thoughts, list) and novel_thoughts:
        rendered = " | ".join(str(item)[:120] for item in novel_thoughts[:2] if item)
        if rendered:
            directives.append(f"Novel thought candidates: {rendered}")
    attractor = frame.get("attractor_state") or {}
    if isinstance(attractor, dict):
        selected = str(attractor.get("selected") or "").strip()
        recurrent_depth = attractor.get("recurrent_depth")
        if selected:
            directives.append(
                f"Attractor state: center the reply on {selected.replace('_', ' ')}"
                + (f" with recurrent_depth={recurrent_depth}." if recurrent_depth else ".")
            )
    working_memory = frame.get("working_memory") or {}
    if isinstance(working_memory, dict):
        admission = str(working_memory.get("admission") or "admit")
        if admission != "admit":
            directives.append(
                f"Working-memory gate: {admission}; keep the response compact and stable while preserving intent."
            )
    causal_effects = frame.get("causal_effects") or {}
    if isinstance(causal_effects, dict):
        attention = causal_effects.get("attention_focus") or []
        if isinstance(attention, list) and attention:
            rendered_attention = ", ".join(str(item)[:40] for item in attention[:4] if item)
            if rendered_attention:
                directives.append(f"Attention targets: {rendered_attention}.")
        memory_priority = _bounded_float(causal_effects.get("memory_priority"), 0.0)
        if memory_priority >= 0.45:
            directives.append("Let the model influence what should be remembered or compared against prior context.")
        verify_pressure = _bounded_float(causal_effects.get("verification_pressure"), 0.0)
        if verify_pressure >= 0.35:
            directives.append("Mark which parts are hypothetical versus verified before acting.")
    if bool(routing.get("seek_verification")):
        directives.append("If the request needs real-world effects or facts, route through governed tools before claiming completion.")
    return " ".join(directives)


def _compact_bicameral_directive(frame: dict[str, Any] | None) -> str:
    if not isinstance(frame, dict):
        return ""
    try:
        from core.brain.bicameral_advisory import validate_bicameral_frame

        if not validate_bicameral_frame(frame):
            return ""
    except (ImportError, RuntimeError, TypeError, ValueError):
        return ""
    try:
        salience = float(frame.get("salience", 0.0) or 0.0)
    except (TypeError, ValueError):
        salience = 0.0
    if salience < 0.18:
        return ""

    routing = frame.get("routing_bias") or {}
    causal = frame.get("causal_effects") or {}
    attention = frame.get("attention_targets") or []
    if not isinstance(routing, dict):
        routing = {}
    if not isinstance(causal, dict):
        causal = {}
    if not isinstance(attention, list):
        attention = []

    directives = [
        "Bicameral advisory: reconcile internal proposals into one coherent answer; do not present them as voices or evidence of phenomenal experience."
    ]
    summary = " ".join(str(frame.get("narrator_summary") or "").split())
    if summary:
        directives.append(summary[:260])
    if routing.get("use_tool_gateway"):
        directives.append("External effects require governed tool execution and post-action evidence.")
    if routing.get("seek_verification"):
        directives.append("Verify before claiming facts, tool completion, or successful file/browser actions.")
    if routing.get("raise_metacognition"):
        directives.append("Check assumptions and resolve uncertainty before answering strongly.")
    if routing.get("use_imagination") or routing.get("expand_options"):
        directives.append("Use a novel option or analogy if it helps the user's actual request.")
    rendered_attention = ", ".join(
        " ".join(str(item).split())[:40] for item in attention[:4] if item
    )
    if rendered_attention:
        directives.append(f"Attention: {rendered_attention}.")
    if _bounded_float(causal.get("memory_priority"), 0.0) >= 0.45:
        directives.append("Preserve continuity with relevant prior conversation or memory.")
    return " ".join(directives)


def _compact_cognitive_situation_directive(frame: dict[str, Any] | None) -> str:
    if not isinstance(frame, dict):
        return ""
    try:
        from core.brain.cognitive_situation import render_cognitive_situation_prompt_block

        return render_cognitive_situation_prompt_block(frame, compact=True).strip()
    except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "cognitive_engine",
            exc,
            severity="warning",
            action="continued desktop quick reply without cognitive situation prompt block",
        )
        logger.debug("Cognitive situation directive unavailable: %s", exc)
        return ""


def _record_the_capability_inventory_miss(
    *,
    capability_inventory_contract: Any,
    system_prompt: Any,
    visible_user_message: Any,
) -> None:
    """Record why the capability inventory contract did not hold.

    Moved out of ``CognitiveEngine._direct_desktop_quick_reply`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 3 name(s) from the turn and hands back
    0.
    """
    if not capability_inventory_contract:
        try:
            from core.brain.present_moment import present_moment_block

            present = present_moment_block()
            if present:
                # PREPENDED, not appended. The system prompt is compacted to
                # a 2,400-char scaffold floor before it reaches the worker,
                # keeping the head and a critical excerpt; anything at the
                # tail is the first thing cut. Attached at the tail, this
                # block was logged as attached and still never arrived —
                # "what time is it?" answered "my clock says 06:15 and the
                # ambient light sensors report low illumination" at 01:40,
                # from a runtime with no light sensor.
                # NOT prepended any more. inference_gate now delivers this
                # same block as its own system message positioned just
                # BEFORE the final user turn — added after compaction, so
                # it cannot be trimmed away, which was the original reason
                # for prepending here.
                #
                # Prepending it a second time put per-turn volatile text
                # (a clock, "2 min ago" receipts) at token ~125 of the
                # system prompt, which invalidated the KV prefix for
                # everything behind it. Measured live: 1,648 of 1,834
                # tokens re-prefilled every turn (10% reuse) and a simple
                # reply taking 13-16s, almost all of it prefill.
                pass
                # Grounding that cannot be seen cannot be verified. Two
                # prompt builders and one of them ungrounded cost an hour
                # of reasoning about why a fix "did not work" when it had
                # simply never run.
                logger.info(
                    "🧭 [GROUNDING] present-moment prepended to the desktop "
                    "system prompt (+%d chars, total %d).",
                    len(present),
                    len(system_prompt),
                )
        except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="continued desktop turn without present-moment grounding",
            )
        try:
            from core.brain.recent_actions import recent_actions_block

            actions = recent_actions_block()
            if actions:
                # Same: the gate places the receipts block before the final
                # user turn. This copy only cost the cache prefix.
                pass
        except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="continued desktop turn without recent-action receipts",
            )
        try:
            # Wider predicate: this path only adds a reading, while
            # asks_about_own_runtime also turns off web search.
            from core.runtime.self_state_intent import (
                asks_about_own_capabilities,
            )

            if asks_about_own_capabilities(visible_user_message):
                from core.brain.self_state_report import runtime_self_report

                instruments = runtime_self_report()
                if instruments:
                    # Same: delivered by the gate.
                    pass
        except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="continued desktop turn without runtime self-readings",
            )


async def _commit_the_thought_with_retries(
    *,
    commit_outcome: Any,
    cycle_deadline_at: Any,
    is_test_run: Any,
    max_retries: Any,
    origin: Any,
    pre_turn_cognition: Any,
    self: Any,
    should_bypass_commit: Any,
    state: Any,
    temp_state: Any,
) -> tuple[Any, Any]:
    """Commit the thought, retrying inside the attempt budget.

    Moved out of ``CognitiveEngine._run_thinking_loop`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 10 name(s) from the turn and hands back
    2.
    """
    from core.state.state_repository import StateVersionConflictError
    for attempt in range(max_retries):
        if should_bypass_commit:
            commit_outcome = (
                "bypassed_test_isolation" if is_test_run else "no_state_repository"
            )
            logger.info("🧠 [STATE] Test run state isolation: bypassing database commit.")
            break
        _commit_budget = max(0.0, cycle_deadline_at - time.monotonic())
        if _commit_budget <= 0.0:
            commit_outcome = "cycle_deadline_expired"
            record_degradation(
                "cognitive_engine",
                TimeoutError("cognitive cycle budget spent before state commit"),
                severity="warning",
                action="skipped the state commit because the cycle deadline had passed",
            )
            break
        try:
            # v14.2: Ensure the repository reference is correct (self.state_repository)
            await asyncio.wait_for(
                self.state_repository.commit(state, "cognitive_cycle"),
                timeout=_commit_budget,
            )
            commit_outcome = "committed"
            break  # Success!
        except TimeoutError:
            commit_outcome = "commit_timeout"
            record_degradation(
                "cognitive_engine",
                TimeoutError(f"state commit exceeded {_commit_budget:.1f}s"),
                severity="error",
                action="abandoned the state commit at the cognitive cycle deadline",
            )
            break
        except StateVersionConflictError as v_err:
            if attempt == max_retries - 1:
                commit_outcome = "version_conflict_exhausted"
                logger.error(
                    "Final state commit failed after %d retries: %s", max_retries, v_err
                )
                break

            logger.warning(
                "🔄 [STATE] Version conflict (attempt %d/%d). Re-deriving from latest...",
                attempt + 1,
                max_retries,
            )
            # Preserve the cognitive work completed in this cycle
            preserved_memory = list(state.cognition.working_memory)
            preserved_objective = state.cognition.current_objective
            preserved_origin = state.cognition.current_origin

            latest = await self.state_repository.get_current()
            state = latest.derive(f"rebase_retry_{attempt + 1}: {origin}", origin=origin)

            # Apply preserved cognitive context onto the newly derived state
            state.cognition.working_memory = preserved_memory
            state.cognition.current_objective = preserved_objective
            state.cognition.current_origin = preserved_origin

            # HF12 Extension: Preserve additional cognitive labor —
            # ONLY the fields this turn actually changed.
            #
            # Copying all of them from the per-turn snapshot overwrote
            # whatever a concurrent writer had committed in the meantime,
            # which is the exact thing a version conflict is telling us
            # happened. A field this turn did not touch keeps the latest
            # value; a field it did touch wins, because that work would
            # otherwise be lost.
            self._reapply_turn_changes(
                state.cognition,
                temp_state.cognition,
                pre_turn_cognition,
            )
        except (RuntimeError, AttributeError, TypeError) as e:
            record_degradation(
                "cognitive_engine",
                e,
                severity="degraded",
                action="stopped commit retry loop and preserved in-memory cognitive result",
            )
            logger.error("Failed to commit final cognitive state: %s", e)
            break
    return commit_outcome, state


def _note_the_quick_reply_contract(
    *,
    ambient_grounding_blocks: Any,
    capability_inventory_contract: Any,
    live_mind_context: Any,
    memory_state_contract: Any,
    mind_context_contract: Any,
    mind_context_lesioned: Any,
    self_condition_contract: Any,
) -> None:
    """Record what the quick-reply contract required of this turn.

    Moved out of ``CognitiveEngine._direct_desktop_quick_reply`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 7 name(s) from the turn and hands back
    0.
    """
    if isinstance(live_mind_context, dict) and live_mind_context and not mind_context_lesioned:
        mind_context_limit = (
            900
            if memory_state_contract
            else 360
            if capability_inventory_contract
            else 700
            if self_condition_contract
            else 2600
        )
        if capability_inventory_contract or self_condition_contract:
            compact_mind_context = {
                "required_for_live_desktop": live_mind_context.get("required_for_live_desktop"),
                "must_answer_from_full_mind_path": live_mind_context.get(
                    "must_answer_from_full_mind_path"
                ),
                "required_subsystems_ok": live_mind_context.get("required_subsystems_ok"),
                "lane": live_mind_context.get("lane"),
                "governance": live_mind_context.get("governance"),
            }
        else:
            compact_mind_context = {
                "required_for_live_desktop": live_mind_context.get("required_for_live_desktop"),
                "must_answer_from_full_mind_path": live_mind_context.get(
                    "must_answer_from_full_mind_path"
                ),
                "required_subsystems_ok": live_mind_context.get("required_subsystems_ok"),
                "required_subsystems": live_mind_context.get("required_subsystems"),
                "lane": live_mind_context.get("lane"),
                "voice": live_mind_context.get("voice"),
                "substrate": live_mind_context.get("substrate"),
                "mind_snapshot": live_mind_context.get("mind_snapshot"),
                "mind_snapshot_quality": live_mind_context.get("mind_snapshot_quality"),
                "governance": live_mind_context.get("governance"),
            }
        live_mind_grounding = (
            "[LIVE MIND CONTEXT]\n"
            f"{_compact_json(compact_mind_context, limit=mind_context_limit)}\n"
            "This is causal grounding for the reply, not text to recite. "
            "If required_for_live_desktop is true, do not answer from a generic assistant persona. "
            "Use the current user turn, the recent role history, memory, substrate, governance, and "
            "inference lane as one live context."
        )
        if mind_context_contract:
            live_mind_grounding = f"{live_mind_grounding}\n{mind_context_contract}"
        ambient_grounding_blocks.append(
            f"{live_mind_grounding}\n[END LIVE MIND CONTEXT]"
        )


class CognitiveEngine:
    """
    Cognitive Engine facade.
    Now delegates to modular phases for structured thinking.
    """

    def __init__(self, backend: Any = None):
        self.backend = backend
        self.thoughts: deque = deque(maxlen=_THOUGHT_HISTORY_LIMIT)
        # Shutdown state. stop() sets this and every cognitive entry point
        # consults it, so a stopped engine cannot keep thinking.
        self._stopped = False
        self._active_tasks: set = set()
        self._phases = []
        self._augmentors = []
        #: Class names of the registered augmentors, for the audit.
        self._augmentor_registry_receipt: list[str] = []
        self.state_repository = None
        self.autopoiesis = AutopoieticGraph()
        self._recovery_lock = RobustLock(
            "CognitiveEngine.RecoveryLock"
        )  # Audit Fix: Mutex for recovery
        self._reasoning: ReasoningStrategies | None = None  # Lazy-init
        #: The router the reasoning layer was built around, so a
        #: replacement or failover rebuilds it instead of being ignored.
        self._reasoning_router: Any = None
        self._reasoning_lock = checked_lock(
            "cognitive_engine.reasoning_layer", rank=LockRank.LEAF
        )

    @property
    def consciousness(self) -> Any:
        """Unified access to the consciousness layer for metric aggregation."""
        return get_container().get("consciousness_core", default=None)

    @property
    def _current_tier(self) -> str:
        """Visibility for routing tests."""
        container = get_container()
        router = container.get("llm_router", default=None)
        if router and hasattr(router, "last_tier"):
            return router.last_tier
        return "unknown"

    @property
    def lobotomized(self) -> bool:
        """True if the engine has no usable cognitive pathway."""
        return self.state_repository is None and len(self._phases) == 0

    def is_ready(self) -> bool:
        """Synchronous liveness probe for user-facing cognition.

        ``lobotomized`` is an AND — no repository AND no phases — so an engine
        with a repository and ZERO phases was ready. There is no cognition
        without phases; that is the whole pipeline.
        """
        return (
            callable(getattr(self, "think", None))
            and isinstance(self.thoughts, deque)
            and getattr(self, "_recovery_lock", None) is not None
            and not self.lobotomized
            and bool(self._phases)
        )

    def setup(self, registry=None, router=None, event_bus=None):
        """Initialize components and phases."""
        container = get_container()
        # Ported Zenith: Phases expect Kernel, but modular boot often passes Container
        # We resolve the kernel instance or use a fallback mechanism
        kernel = container.get("aura_kernel", default=None)

        phase_entries = instantiate_legacy_runtime_phases(
            kernel or container,
            include_executive_closure=False,
        )
        self._phases = [phase for _, phase in phase_entries]

        # ISSUE-97: AuraPipeline Awareness
        #
        # required_phases used to be len(phase_entries) — the length of the
        # very list that populated _phases — so the comparison below was
        # `len(x) != len(x)` and could never fire, including when both were
        # zero. The DECLARED spectrum is what the blueprint says it is.
        required_phases = len(
            legacy_runtime_phase_specs(include_executive_closure=False)
        )
        self._pipeline_receipt = {
            "declared_phases": required_phases,
            "instantiated_phases": len(self._phases),
            "complete": len(self._phases) == required_phases and required_phases > 0,
            "at": time.time(),
        }
        if len(self._phases) != required_phases:
            logger.warning(
                "⚠️ AuraPipeline: Incomplete cognitive pipeline (%d/%d phases).",
                len(self._phases),
                required_phases,
            )
        else:
            logger.info(
                "🧠 AuraPipeline: Full cognitive spectrum online (%d phases).", required_phases
            )

        self.phase_map = {phase.__class__.__name__: phase for _, phase in phase_entries}

    async def on_start_async(self):
        """Lifecycle hook."""
        self.setup()
        logger.info("⚡ CognitiveEngine active.")

    @staticmethod
    def _reapply_turn_changes(target: Any, worked: Any, baseline: dict[str, Any]) -> None:
        """Put back what THIS turn changed, and nothing else.

        ``baseline`` is the value each field held before the phases ran. A
        field the turn left alone keeps whatever the rebased state carries —
        which, after a version conflict, is a concurrent writer's work.
        """
        for field, before in baseline.items():
            after = getattr(worked, field, None)
            if field in {"active_goals", "pending_initiatives"}:
                after_list = list(after or [])
                if after_list != list(before or []):
                    setattr(target, field, after_list)
                continue
            if field == "modifiers":
                after_map = dict(after or {})
                if after_map != dict(before or {}):
                    setattr(target, field, after_map)
                continue
            if after != before:
                setattr(target, field, after)

    @staticmethod
    def _turn_response_message(
        working_memory: Any, *, mark: int
    ) -> dict[str, Any] | None:
        """The assistant message THIS turn produced, or None.

        Extraction used to take the last message if its role was "assistant".
        A duplicate or suppressed user append leaves an older assistant message
        at the end of working memory, and that previous answer went out again
        as this turn's — correct-looking, addressed to the wrong question.
        """
        memory = list(working_memory or [])
        if len(memory) <= max(0, int(mark)):
            return None
        last = memory[-1]
        if not isinstance(last, dict):
            return None
        if str(last.get("role", "") or "").strip().lower() != "assistant":
            return None
        return last

    @staticmethod
    def _cycle_confidence(*, commit_outcome: str, degraded_subsystems: int) -> float:
        """Confidence from what the cycle can show, not a constant.

        0.9 was returned for every successful phase cycle regardless of whether
        durable state committed, whether subsystems were degraded, or whether
        anything validated the response. It still is not a calibrated
        probability — nothing here measures correctness — but it now moves with
        the evidence the cycle actually has, and the floor is what a cycle that
        produced text but could not persist it deserves.
        """
        confidence = 0.9
        if commit_outcome not in {"committed", "bypassed_test_isolation"}:
            # The answer exists; the record of the turn that produced it does
            # not. Downstream retry and learning both need to see that.
            confidence -= 0.25
        confidence -= min(0.2, 0.05 * max(0, int(degraded_subsystems)))
        return round(max(0.3, min(0.95, confidence)), 3)

    def phase_rollback_receipt(self) -> dict[str, Any]:
        """What the last phase-failure rollback restored, and what it could not."""
        return dict(getattr(self, "_last_phase_rollback", {}) or {})

    def pipeline_receipt(self) -> dict[str, Any]:
        """What setup() built, against what the blueprint declares."""
        return dict(getattr(self, "_pipeline_receipt", {}) or {})

    async def check_health(self) -> dict[str, Any]:
        """Health from evidence, not from being callable.

        This returned ``"healthy"`` unconditionally — a zero-phase engine with
        no repository reported a full cognitive spectrum, and every consumer of
        this health believed it.
        """
        receipt = self.pipeline_receipt()
        declared = int(receipt.get("declared_phases", 0) or 0)
        built = len(self._phases)
        reasons: list[str] = []
        if not built:
            reasons.append("no_phases_instantiated")
        elif declared and built != declared:
            reasons.append(f"incomplete_pipeline:{built}/{declared}")
        if self.state_repository is None:
            reasons.append("no_state_repository")
        if self._stopped:
            reasons.append("engine_stopped")
        status = "healthy"
        if not built or self._stopped:
            status = "unhealthy"
        elif reasons:
            status = "degraded"
        return {
            "status": status,
            "reasons": reasons,
            "modular": True,
            "phases_count": built,
            "declared_phases_count": declared,
            "augmentors_count": len(self._augmentors),
            "ready": self.is_ready(),
        }

    #: What an augmentor's output may contribute to one turn. It lands in the
    #: prompt context, so it is bounded like any other prompt material rather
    #: than by whatever the augmentor felt like returning.
    _AUGMENTATION_CHAR_LIMIT = 4_000
    #: An augmentor runs synchronously, on the event loop, before the phases.
    #: A slow one used to hold every turn behind it.
    _AUGMENTATION_TIMEOUT_S = 2.0

    def register_augmentor(self, augmentor: Any) -> bool:
        """Register a cognitive augmentor (e.g. SovereignWebAugmentor).

        This accepted any object at all: no declared name, no callable
        contract, no bound on what it returns — and its output goes into the
        prompt. The admission below is small on purpose (this is an in-process
        extension point, not a plugin marketplace) but it is a contract rather
        than a shrug, and a refusal is recorded instead of silently producing
        an augmentor that raises AttributeError on every turn.
        """
        getter = getattr(augmentor, "get_augmentation", None)
        if not callable(getter):
            record_degradation(
                "cognitive_engine",
                TypeError(
                    f"augmentor {type(augmentor).__name__} has no callable get_augmentation"
                ),
                severity="warning",
                action="refused to register an augmentor with no augmentation contract",
            )
            return False
        if augmentor in self._augmentors:
            return True
        self._augmentors.append(augmentor)
        self._augmentor_registry_receipt = [
            type(existing).__name__ for existing in self._augmentors
        ]
        logger.info("🧠 CognitiveEngine: Registered augmentor %s", type(augmentor).__name__)
        return True

    def augmentor_registry_receipt(self) -> list[str]:
        """Which augmentors may contribute to a turn."""
        return list(getattr(self, "_augmentor_registry_receipt", []) or [])

    #: What a caller-supplied contract may contribute to the system prompt.
    #: The same limits the inference gate applies to the same two strings —
    #: they were bounded there and concatenated raw here.
    _STYLE_CONTRACT_LIMIT = 1_400
    #: What a vision query may be. It reaches a visual model with an image
    #: attached, so it is bounded and cannot carry contract structure.
    _VISION_QUERY_LIMIT = 600
    _MIND_CONTRACT_LIMIT = 900

    @staticmethod
    def _bounded_request_int(raw: Any, *, default: int, low: int, high: int) -> int:
        """An integer from caller context, or the default. Never a raise."""
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return default
        if value != value or value in (float("inf"), float("-inf")):
            return default
        return max(low, min(high, int(value)))

    @staticmethod
    def _log_safe_objective(objective: Any, limit: int = 50) -> str:
        """A log-safe preview of what was asked.

        Scrubbed, not just truncated: truncation preserves the FIRST fifty
        characters, which is exactly where an address, a key or a phone number
        appears in a message that opens with one.
        """
        text = str(objective or "")
        try:
            from core.brain.pii_scrubber import scrub_pii_for_cloud

            text = scrub_pii_for_cloud(text)
        except (ImportError, RuntimeError, TypeError, ValueError):
            # Unable to scrub is not permission to print.
            return "[objective unavailable for logging]"
        return text[:limit]

    @staticmethod
    def _contract_safe(value: Any, limit: int) -> str:
        """Flatten a caller string so it cannot forge system-prompt structure.

        response_style_contract and mind_context_contract arrive in the
        caller's context and were concatenated straight into the system
        message. A line break turns the rest into a sibling instruction; a
        leading "#" opens a sibling section; a chat control token forges a role
        boundary. The inference gate already neutralizes both of these strings
        (_contract_safe there); this path did not, so the same value was safe
        through one door and privileged through the other.
        """
        from core.brain.living_mind_context import neutralize_learned_text

        text = neutralize_learned_text(str(value or ""))
        if not text:
            return ""
        text = " ".join(text.split())
        if len(text) <= limit:
            return text
        return text[: max(1, limit - 1)].rstrip() + "…"

    @classmethod
    def _bounded_augmentation(cls, raw: Any) -> Any:
        """Bound and neutralize what an augmentor contributes to the prompt.

        Augmentor output is not one of Aura's own measurements — it is
        whatever a registered object returned — and it reaches the prompt. It
        is bounded, and its text cannot forge contract structure.
        """
        from core.brain.living_mind_context import neutralize_learned_text

        if isinstance(raw, str):
            return neutralize_learned_text(raw)[: cls._AUGMENTATION_CHAR_LIMIT]
        if isinstance(raw, dict):
            return {
                str(key)[:120]: cls._bounded_augmentation(value)
                for key, value in list(raw.items())[:32]
            }
        if isinstance(raw, (list, tuple)):
            return [cls._bounded_augmentation(item) for item in list(raw)[:32]]
        return raw

    @staticmethod
    def _normalize_mode(mode: ThinkingMode | str | Any) -> ThinkingMode:
        if isinstance(mode, ThinkingMode):
            return mode
        if isinstance(mode, str):
            normalized = mode.strip().lower()
            for candidate in ThinkingMode:
                if candidate.name.lower() == normalized:
                    return candidate
        return ThinkingMode.FAST

    @classmethod
    def _is_background_request(cls, origin: str, explicit_background: bool) -> bool:
        return background_policy.is_background_origin(
            origin, explicit_background=explicit_background
        )

    @staticmethod
    def _empty_thought(mode: ThinkingMode, reason: str) -> Thought:
        return Thought(
            id=str(uuid.uuid4()),
            content="",
            mode=mode,
            confidence=0.0,
            reasoning=[reason],
            metadata={"suppressed": True},
        )

    def _should_suppress_background_reflection(
        self, mode: ThinkingMode, is_background: bool
    ) -> bool:
        if not is_background or mode not in _BACKGROUND_REFLECTIVE_MODES:
            return False

        try:
            container = get_container()
            orchestrator = container.get("orchestrator", default=None)
            if orchestrator:
                status = getattr(orchestrator, "status", None)
                if status and getattr(status, "is_processing", False):
                    return True

                last_user = float(getattr(orchestrator, "_last_user_interaction_time", 0.0) or 0.0)
                if last_user and (time.time() - last_user) < 180.0:
                    return True
        except (OSError, ConnectionError, TimeoutError) as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="continued without orchestration-based background suppression",
            )
            logger.debug("Background reflection suppression check failed: %s", exc)

        try:
            from core.runtime import resource_psutil as psutil

            if psutil.virtual_memory().percent >= 80.0:
                return True
        except (ImportError, AttributeError, RuntimeError) as _exc:
            record_degradation(
                "cognitive_engine",
                _exc,
                severity="warning",
                action="continued without memory-pressure background suppression",
            )
            logger.debug("Suppressed Exception: %s", _exc)

        return False

    def _background_suppression_reason(self) -> str:
        try:
            container = get_container()
            orchestrator = container.get("orchestrator", default=None)
            if orchestrator is None:
                return ""
            return str(
                background_policy.background_activity_reason(
                    orchestrator,
                    profile=background_policy.THOUGHT_BACKGROUND_POLICY,
                )
                or ""
            )
        except (OSError, ConnectionError, TimeoutError) as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="returned empty background suppression reason",
            )
            logger.debug("Background thought policy check failed: %s", exc)
            return ""

    async def _set_recovery_in_progress(self, value: bool) -> bool:
        """Flip the recovery flag under a short lock. Returns whether it took.

        CP126 45d7e755. On lock-acquisition failure this wrote the shared
        flag anyway — defeating the mutex precisely under contention, which
        is the only time it matters, and letting overlapping recoveries set
        or CLEAR each other's state.

        The two directions are not symmetric, so they are not treated the
        same:

        * Setting True unsynchronised can only over-mark. Failing to acquire
          usually means someone else holds the lock — i.e. a recovery really
          is in progress — so the write agrees with reality and is allowed.
        * Clearing to False unsynchronised can erase a recovery another task
          still owns, so it is refused. A flag left set resolves when the
          owning task clears it; a flag wrongly cleared invites a second
          concurrent recovery.
        """
        if await self._recovery_lock.acquire_robust(timeout=1.0):
            try:
                self._recovery_in_progress = value
            finally:
                if self._recovery_lock.locked():
                    self._recovery_lock.release()
            return True

        if value:
            self._recovery_in_progress = True
            record_degradation(
                "cognitive_engine",
                RuntimeError("recovery lock unavailable while marking recovery active"),
                severity="warning",
                action="set the recovery flag unsynchronised; over-marking is safe",
            )
            return True

        record_degradation(
            "cognitive_engine",
            RuntimeError("recovery lock unavailable while clearing recovery state"),
            severity="warning",
            action=(
                "refused to clear the recovery flag without the lock; another "
                "recovery may still own it"
            ),
        )
        return False

    async def generate_autonomous_thought(self, prompt: str = None, **kwargs) -> Thought:
        """Entry point for self-initiated/autonomous thinking."""
        objective = prompt or "Reflecting on current inner state and environment."
        return await self.think(objective, origin="autonomous", **kwargs)

    @staticmethod
    def _normalize_origin(origin: Any) -> str:
        return normalize_objective_origin(origin)

    @classmethod
    def _is_user_facing_origin(cls, origin: Any) -> bool:
        return is_foreground_objective_origin(origin)

    #: Reported once per (origin, source): a caller that omits its origin on
    #: every request would otherwise fill the trail.
    _refused_origin_inheritance_seen: set[tuple[str, str]] = set()

    @classmethod
    def _note_refused_origin_inheritance(cls, origin: str, *, source: str) -> None:
        key = (str(origin), str(source))
        if key in cls._refused_origin_inheritance_seen:
            return
        cls._refused_origin_inheritance_seen.add(key)
        logger.warning(
            "🛡️ Refused to inherit the user-facing origin %r from %s: a request "
            "that did not declare its own principal does not get one.",
            origin,
            source,
        )

    @classmethod
    def refused_origin_inheritances(cls) -> list[tuple[str, str]]:
        """(origin, source) pairs this process declined to inherit."""
        return sorted(cls._refused_origin_inheritance_seen)

    @classmethod
    def _is_test_run(cls, origin: Any) -> bool:
        """Whether THIS request runs under test isolation.

        AURA_TESTING and AURA_AGI_MAX_TASKS are process-wide, and this used to
        read them for every origin — so in a mixed process (a suite running
        beside the live runtime, or a developer with the variable exported) a
        real person's turn silently got a substituted default state and its
        commit bypassed. Their answer would be produced from no memory and
        remembered by nothing.

        Ambient variables still mark test runs; they just cannot make a
        USER-FACING turn into one. A live turn keeps its state and its commit
        whatever the environment says.
        """
        if str(origin or "").strip().lower() == "test":
            return True
        ambient = (
            env_present(
                "AURA_AGI_MAX_TASKS",
                description="Bounded AGI proof task-count override",
                owner="core.brain.cognitive_engine",
            )
            or env_present(
                "AURA_TESTING",
                description="Process test-isolation marker",
                owner="core.brain.cognitive_engine",
            )
        )
        if not ambient:
            return False
        if cls._is_user_facing_origin(origin):
            cls._note_refused_test_isolation(str(origin or ""))
            return False
        return True

    _refused_test_isolation_seen: set[str] = set()

    @classmethod
    def _note_refused_test_isolation(cls, origin: str) -> None:
        if origin in cls._refused_test_isolation_seen:
            return
        cls._refused_test_isolation_seen.add(origin)
        logger.warning(
            "🛡️ Ambient test environment is set, but origin %r is user-facing: "
            "keeping real state and committing this turn.",
            origin,
        )

    @classmethod
    def _resolve_origin(cls, origin: Any, context: dict[str, Any] | None = None) -> str:
        normalized = cls._normalize_origin(origin)
        if normalized:
            return normalized

        if isinstance(context, dict):
            for key in ("origin", "request_origin", "intent_source"):
                contextual = cls._normalize_origin(context.get(key))
                if contextual:
                    return contextual

        # Below here the origin comes from SHARED MUTABLE STATE — the
        # orchestrator's last-seen origin and the repository's latest state.
        # Two requests in flight can read each other's, and inheriting a
        # USER-FACING origin is a privilege escalation: it grants the protected
        # Cortex lane, trust treatment, and foreground admission to a request
        # whose caller never claimed to be a person. Inheriting a background
        # origin costs nothing, so only that inheritance is allowed.
        try:
            container = get_container()
            orchestrator = container.get("orchestrator", default=None)
            orchestrator_origin = cls._normalize_origin(
                getattr(orchestrator, "_current_origin", "")
            )
            if orchestrator_origin:
                if cls._is_user_facing_origin(orchestrator_origin):
                    cls._note_refused_origin_inheritance(
                        orchestrator_origin, source="orchestrator"
                    )
                else:
                    return orchestrator_origin

            repo = container.get("state_repository", default=None)
            live_state = getattr(repo, "_current", None) if repo is not None else None
            state_origin = cls._normalize_origin(
                getattr(getattr(live_state, "cognition", None), "current_origin", "")
            )
            if state_origin:
                if cls._is_user_facing_origin(state_origin):
                    cls._note_refused_origin_inheritance(
                        state_origin, source="state_repository"
                    )
                else:
                    return state_origin
        except (OSError, ConnectionError, TimeoutError) as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="defaulted unresolved cognitive origin to system",
            )
            logger.debug("CognitiveEngine origin resolution degraded: %s", exc)

        return "system"

    def _apply_spiking_active_inference(
        self,
        state: AuraState,
        objective: str,
        origin: str,
        context: dict[str, Any] | None,
        *,
        is_background: bool,
    ) -> dict[str, Any] | None:
        try:
            from core.cognitive.spiking_active_inference import (
                get_spiking_active_inference_advisor,
            )

            advisor = get_spiking_active_inference_advisor()
            advice = advisor.advise(
                objective,
                context=context,
                state=state,
                origin=origin,
                is_background=is_background,
            )
        except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="continued cognitive cycle without spiking active-inference advisory",
            )
            logger.debug("Spiking active-inference advisory unavailable: %s", exc)
            return context

        advice_dict = advice.to_dict()
        routing = dict(advice.routing_bias or {})
        sampling = dict(advice.sampling_bias or {})
        state.response_modifiers["spiking_active_inference"] = advice_dict
        state.response_modifiers["active_inference_action_tendency"] = advice.action
        state.response_modifiers["epistemic_uncertainty"] = advice.uncertainty
        state.response_modifiers["metacognition_depth"] = routing.get("metacognition_depth", 0.35)
        state.response_modifiers["tool_governance_pressure"] = bool(
            routing.get("use_tool_gateway")
        )
        state.response_modifiers["sampling_bias"] = sampling
        if routing.get("reduce_load"):
            state.response_modifiers["runtime_load_shed_requested"] = True
        if routing.get("repair_first"):
            state.response_modifiers["repair_first_pressure"] = True

        merged_context = dict(context or {})
        merged_context["spiking_active_inference"] = advice_dict
        return merged_context

    def _apply_entity_memory(
        self,
        state: AuraState,
        objective: str,
        context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Make what Aura knows about the people/places/things in play causal.

        This runs on the live path so recognising an entity actually changes
        retrieval depth, retrieval targeting, and affect before the answer is
        generated — see core/memory/entity_memory_bridge.py, which owns the
        effects. Failure here is never fatal: the turn proceeds without the
        entity context, which is exactly how it behaved before this existed.
        """
        merged_context = dict(context or {})
        try:
            from core.memory.entity_memory_bridge import apply_entity_context

            # source="user": the objective is what the interlocutor asked, so
            # it may introduce entities. Aura's own generated text is never
            # passed here — a name she invented must not become a permanent
            # member of her world that later mentions then "confirm".
            summary = apply_entity_context(
                state, objective, merged_context,
                source="user",
                evidence_id=str(merged_context.get("evidence_id") or ""),
            )
            if summary.get("entities"):
                merged_context["entity_memory"] = (
                    summary.get("context", {}).get("entity_memory")
                    or merged_context.get("entity_memory")
                )
                state.cognition.modifiers["entity_memory_effects"] = list(
                    summary.get("effects", [])
                )
                logger.debug(
                    "🧠 Entity memory: %d entity(ies) in play, %d effect(s).",
                    len(summary["entities"]), len(summary.get("effects", [])),
                )
        except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="continued cognitive cycle without entity memory context",
            )
            logger.debug("Entity memory context skipped: %s", exc)
        return merged_context

    def _apply_imagination_workspace(
        self,
        state: AuraState,
        objective: str,
        origin: str,
        context: dict[str, Any] | None,
        *,
        is_background: bool,
    ) -> dict[str, Any] | None:
        try:
            from core.brain.imagination import get_imagination_engine

            engine = get_imagination_engine()
            frame = engine.imagine(
                objective,
                state=state,
                context=context,
                origin=origin,
                is_background=is_background,
            )
        except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="continued cognitive cycle without imagination workspace",
            )
            logger.debug("Imagination workspace unavailable: %s", exc)
            return context

        frame_dict = frame.to_dict()
        if frame.salience < 0.18:
            return context

        state.response_modifiers["imagination_workspace"] = frame_dict
        state.response_modifiers["creative_pressure"] = frame.salience
        state.response_modifiers["novelty_pressure"] = frame.novelty_pressure
        state.response_modifiers["imagination_sampling_bias"] = dict(frame.sampling_bias)
        state.response_modifiers["imagination_routing_bias"] = dict(frame.routing_bias)
        state.response_modifiers["imagination_memory_pressure"] = frame.memory_pressure
        state.response_modifiers["imagination_verification_pressure"] = frame.verification_pressure
        state.response_modifiers["imagination_working_memory"] = dict(frame.working_memory)
        state.response_modifiers["imagination_attractor_state"] = dict(frame.attractor_state)
        state.response_modifiers["verification_pressure"] = max(
            _bounded_float(state.response_modifiers.get("verification_pressure"), 0.0),
            frame.verification_pressure,
        )
        if frame.routing_bias.get("seek_verification") or frame.routing_bias.get("raise_metacognition"):
            state.response_modifiers["tool_governance_pressure"] = True
            state.response_modifiers["metacognition_depth"] = max(
                _bounded_float(state.response_modifiers.get("metacognition_depth"), 0.35),
                _bounded_float(frame.causal_effects.get("metacognition_depth"), 0.35),
            )

        cognition_mods = dict(getattr(state.cognition, "modifiers", {}) or {})
        cognition_mods["imagination_workspace"] = frame_dict
        cognition_mods["imagination_prompt_block_available"] = True
        cognition_mods["imagination_attention_targets"] = list(frame.attention_targets)
        cognition_mods["imagination_causal_effects"] = dict(frame.causal_effects)
        cognition_mods["imagination_ablation_predictions"] = dict(frame.ablation_predictions)
        cognition_mods["imagination_working_memory"] = dict(frame.working_memory)
        cognition_mods["imagination_attractor_state"] = dict(frame.attractor_state)
        if frame.routing_bias.get("requires_memory_grounding"):
            cognition_mods["requires_memory_grounding"] = True
        if frame.routing_bias.get("compress_imagination"):
            state.response_modifiers["runtime_load_shed_requested"] = True
            cognition_mods["runtime_load_shed_requested"] = True
        state.cognition.modifiers = cognition_mods
        if frame.attention_targets and not is_background:
            state.cognition.attention_focus = (
                f"{objective[:120]} | imagined focus: {', '.join(frame.attention_targets[:3])}"
            )

        merged_context = dict(context or {})
        merged_context["imagination_workspace"] = frame_dict
        return merged_context

    def _apply_bicameral_advisory(
        self,
        state: AuraState,
        objective: str,
        origin: str,
        context: dict[str, Any] | None,
        *,
        is_background: bool,
    ) -> dict[str, Any] | None:
        try:
            from core.brain.bicameral_advisory import get_bicameral_advisory

            advisor = get_bicameral_advisory()
            frame = advisor.advise(
                objective,
                state=state,
                context=context,
                origin=origin,
                is_background=is_background,
            )
        except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="continued cognitive cycle without bicameral advisory",
            )
            logger.debug("Bicameral advisory unavailable: %s", exc)
            return context

        if frame.salience < 0.18:
            return context

        frame_dict = frame.to_dict()
        # The issued frame is deeply immutable. AuraState is intentionally
        # deepcopy-able for phase retry/rebase, so only its fully materialized
        # signed transport payload may cross into state modifiers.
        causal = dict(frame_dict.get("causal_effects") or {})
        routing = dict(frame_dict.get("routing_bias") or {})
        sampling = dict(frame_dict.get("sampling_bias") or {})

        state.response_modifiers["bicameral_advisory"] = frame_dict
        state.response_modifiers["bicameral_consensus"] = frame.consensus
        state.response_modifiers["bicameral_dissent"] = frame.dissent
        state.response_modifiers["bicameral_sampling_bias"] = sampling
        state.response_modifiers["bicameral_routing_bias"] = routing
        state.response_modifiers["bicameral_attention_targets"] = list(frame.attention_targets)
        state.response_modifiers["bicameral_causal_effects"] = causal
        state.response_modifiers["bicameral_memory_priority"] = _bounded_float(
            causal.get("memory_priority"), 0.0
        )
        state.response_modifiers["bicameral_verification_pressure"] = _bounded_float(
            causal.get("verification_pressure"), 0.0
        )
        state.response_modifiers["self_model_update_pressure"] = max(
            _bounded_float(state.response_modifiers.get("self_model_update_pressure"), 0.0),
            _bounded_float(causal.get("self_model_update"), 0.0),
        )
        state.response_modifiers["metacognition_depth"] = max(
            _bounded_float(state.response_modifiers.get("metacognition_depth"), 0.35),
            _bounded_float(causal.get("metacognition_depth"), 0.35),
        )
        state.response_modifiers["verification_pressure"] = max(
            _bounded_float(state.response_modifiers.get("verification_pressure"), 0.0),
            _bounded_float(causal.get("verification_pressure"), 0.0),
        )
        state.response_modifiers["creative_pressure"] = max(
            _bounded_float(state.response_modifiers.get("creative_pressure"), 0.0),
            _bounded_float(causal.get("creative_pressure"), 0.0),
        )
        if routing.get("use_tool_gateway") or routing.get("seek_verification"):
            state.response_modifiers["tool_governance_pressure"] = True
        if routing.get("compact_foreground"):
            state.response_modifiers["runtime_load_shed_requested"] = True
        if (
            _bounded_float(causal.get("memory_priority"), 0.0) >= 0.45
            or _bounded_float(causal.get("self_model_update"), 0.0) >= 0.35
            or routing.get("preserve_conversation_context")
        ):
            state.response_modifiers["requires_memory_grounding"] = True

        cognition_mods = dict(getattr(state.cognition, "modifiers", {}) or {})
        cognition_mods["bicameral_advisory"] = frame_dict
        cognition_mods["bicameral_prompt_block_available"] = True
        cognition_mods["bicameral_attention_targets"] = list(frame.attention_targets)
        cognition_mods["bicameral_causal_effects"] = causal
        cognition_mods["bicameral_sampling_bias"] = sampling
        cognition_mods["bicameral_routing_bias"] = routing
        cognition_mods["self_model_update_pressure"] = state.response_modifiers[
            "self_model_update_pressure"
        ]
        if state.response_modifiers.get("requires_memory_grounding"):
            cognition_mods["requires_memory_grounding"] = True
        state.cognition.modifiers = cognition_mods

        if frame.attention_targets and not is_background:
            existing_focus = str(getattr(state.cognition, "attention_focus", "") or "").strip()
            advisory_focus = ", ".join(frame.attention_targets[:4])
            state.cognition.attention_focus = (
                f"{existing_focus} | advisory focus: {advisory_focus}"
                if existing_focus
                else f"{objective[:120]} | advisory focus: {advisory_focus}"
            )

        merged_context = dict(context or {})
        merged_context["bicameral_advisory"] = frame_dict
        merged_context["bicameral_sampling_bias"] = sampling
        return merged_context

    def _apply_cognitive_situation_frame(
        self,
        state: AuraState,
        objective: str,
        origin: str,
        context: dict[str, Any] | None,
        *,
        is_background: bool,
    ) -> dict[str, Any] | None:
        try:
            from core.brain.cognitive_situation import get_cognitive_situation_engine

            engine = get_cognitive_situation_engine()
            frame = engine.frame(
                objective,
                state=state,
                context=context,
                origin=origin,
                is_background=is_background,
            )
        except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="continued cognitive cycle without cognitive situation frame",
            )
            logger.debug("Cognitive situation frame unavailable: %s", exc)
            return context

        frame_dict = frame.to_dict()
        if frame.salience < 0.16:
            return context

        causal = dict(frame.causal_effects or {})
        routing = dict(frame.routing_bias or {})
        sampling = dict(frame.sampling_bias or {})

        state.response_modifiers["cognitive_situation_frame"] = frame_dict
        state.response_modifiers["semantic_flexibility_pressure"] = frame.semantic_flexibility
        state.response_modifiers["analogical_leap_pressure"] = frame.analogical_leap_pressure
        state.response_modifiers["sensorimotor_grounding_pressure"] = frame.sensorimotor_grounding
        state.response_modifiers["cognitive_situation_sampling_bias"] = sampling
        state.response_modifiers["cognitive_situation_routing_bias"] = routing
        state.response_modifiers["cognitive_situation_attention_targets"] = list(
            frame.attention_targets
        )
        state.response_modifiers["verification_pressure"] = max(
            _bounded_float(state.response_modifiers.get("verification_pressure"), 0.0),
            frame.verification_pressure,
        )
        state.response_modifiers["metacognition_depth"] = max(
            _bounded_float(state.response_modifiers.get("metacognition_depth"), 0.35),
            frame.metacognition_pressure,
        )
        state.response_modifiers["creative_pressure"] = max(
            _bounded_float(state.response_modifiers.get("creative_pressure"), 0.0),
            frame.analogical_leap_pressure,
        )
        if routing.get("use_tool_gateway") or routing.get("bind_sensorimotor_evidence"):
            state.response_modifiers["tool_governance_pressure"] = True
        if routing.get("perception_abstention_required"):
            state.response_modifiers["perception_abstention_required"] = True
        if routing.get("perception_repair_required"):
            state.response_modifiers["perception_repair_required"] = True
        perception_constraints = causal.get("perception_planning_constraints")
        if isinstance(perception_constraints, list):
            state.response_modifiers["perception_planning_constraints"] = list(
                perception_constraints[:8]
            )
        perception_repairs = causal.get("perception_repair_requirements")
        if isinstance(perception_repairs, list):
            state.response_modifiers["perception_repair_requirements"] = list(
                perception_repairs[:8]
            )
        social_constraints = causal.get("social_planning_constraints")
        if isinstance(social_constraints, list):
            state.response_modifiers["social_planning_constraints"] = list(
                social_constraints[:8]
            )
        state.response_modifiers["social_uncertainty"] = frame.social_uncertainty
        state.response_modifiers["social_repair_pressure"] = frame.social_repair_pressure
        if routing.get("social_repair_required"):
            state.response_modifiers["social_repair_required"] = True
        if routing.get("social_confirmation_required"):
            state.response_modifiers["social_confirmation_required"] = True
        if routing.get("social_state_clarification_required"):
            state.response_modifiers["social_state_clarification_required"] = True
        if routing.get("social_response_brevity"):
            state.response_modifiers["social_response_brevity"] = True
        if routing.get("requires_memory_grounding") or routing.get("preserve_conversation_context"):
            state.response_modifiers["requires_memory_grounding"] = True
        if routing.get("deliberate_mode") and not is_background:
            state.cognition.current_mode = CognitiveMode.DELIBERATE

        cognition_mods = dict(getattr(state.cognition, "modifiers", {}) or {})
        cognition_mods["cognitive_situation_frame"] = frame_dict
        cognition_mods["cognitive_situation_prompt_block_available"] = True
        cognition_mods["semantic_flexibility_pressure"] = frame.semantic_flexibility
        cognition_mods["analogical_leap_pressure"] = frame.analogical_leap_pressure
        cognition_mods["sensorimotor_grounding_pressure"] = frame.sensorimotor_grounding
        cognition_mods["cognitive_situation_sampling_bias"] = sampling
        cognition_mods["cognitive_situation_routing_bias"] = routing
        cognition_mods["cognitive_situation_causal_effects"] = causal
        if routing.get("requires_memory_grounding"):
            cognition_mods["requires_memory_grounding"] = True
        if routing.get("bind_sensorimotor_evidence"):
            cognition_mods["bind_sensorimotor_evidence"] = True
        if routing.get("perception_abstention_required"):
            cognition_mods["perception_abstention_required"] = True
        if routing.get("perception_repair_required"):
            cognition_mods["perception_repair_required"] = True
        if routing.get("social_repair_required"):
            cognition_mods["social_repair_required"] = True
        if routing.get("social_confirmation_required"):
            cognition_mods["social_confirmation_required"] = True
        if routing.get("social_state_clarification_required"):
            cognition_mods["social_state_clarification_required"] = True
        state.cognition.modifiers = cognition_mods

        if frame.attention_targets and not is_background:
            existing_focus = str(getattr(state.cognition, "attention_focus", "") or "").strip()
            situation_focus = ", ".join(frame.attention_targets[:4])
            state.cognition.attention_focus = (
                f"{existing_focus} | situation focus: {situation_focus}"
                if existing_focus
                else f"{objective[:120]} | situation focus: {situation_focus}"
            )

        merged_context = dict(context or {})
        merged_context["cognitive_situation_frame"] = frame_dict
        merged_context["cognitive_situation_sampling_bias"] = sampling
        return merged_context

    def _learn_spiking_active_inference_outcome(
        self,
        context: dict[str, Any] | None,
        *,
        outcome: str,
        reward: float,
    ) -> dict[str, Any] | None:
        if not isinstance(context, dict):
            return None
        advice = context.get("spiking_active_inference")
        if not isinstance(advice, dict):
            return None
        action = str(advice.get("action") or "").strip()
        features = advice.get("features")
        if not action or not isinstance(features, dict):
            return None
        try:
            advisor = get_container().get("spiking_active_inference", default=None)
            if advisor is None or not hasattr(advisor, "learn_from_feedback"):
                return None
            learned = advisor.learn_from_feedback(action, float(reward), features)
            if isinstance(learned, dict):
                learned["outcome"] = str(outcome or "unknown")[:80]
                return learned
        except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="continued cognitive cycle without spiking active-inference feedback learning",
            )
            logger.debug("Spiking active-inference feedback learning skipped: %s", exc)
        return None

    def _learn_imagination_workspace_outcome(
        self,
        context: dict[str, Any] | None,
        *,
        outcome: str,
        reward: float,
        evidence_basis: str = "",
        evidence_id: str = "",
    ) -> dict[str, Any] | None:
        if not isinstance(context, dict):
            return None
        frame = context.get("imagination_workspace")
        if not isinstance(frame, dict):
            return None
        try:
            from core.brain.imagination_basis import Basis, meets

            try:
                basis = Basis(str(evidence_basis or ""))
            except ValueError:
                basis = Basis.LEXICAL
            frame_id = str(frame.get("frame_id") or "")[:120]
            subject = str(
                context.get("user_id")
                or context.get("principal_id")
                or "anonymous"
            )[:64]
            if not evidence_id or not meets(basis, Basis.MEASURED):
                # A generation existing is not evidence that imagination made
                # it correct. Keep the eligibility record typed and pending;
                # do not call the durable learner until an evaluator, tool
                # receipt, or user outcome supplies measured evidence.
                return {
                    "frame_id": frame_id,
                    "subject": subject,
                    "outcome": str(outcome or "unknown")[:80],
                    "reward": round(max(-1.0, min(1.0, float(reward))), 4),
                    "evidence_basis": basis.value,
                    "evidence_id": str(evidence_id or "")[:120],
                    "applied": False,
                    "refusal": "measured outcome evidence required",
                }
            from core.brain.imagination import get_imagination_engine

            learned = get_imagination_engine().learn_from_feedback(
                frame,
                reward=float(reward),
                outcome=outcome,
                subject=subject,
                evidence_basis=basis.value,
                evidence_id=evidence_id,
            )
            return learned if isinstance(learned, dict) else None
        except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="continued cognitive cycle without imagination workspace feedback learning",
            )
            logger.debug("Imagination workspace feedback learning skipped: %s", exc)
        return None

    def _learn_bicameral_advisory_outcome(
        self,
        context: dict[str, Any] | None,
        *,
        outcome: str,
        reward: float,
    ) -> dict[str, Any] | None:
        if not isinstance(context, dict):
            return None
        frame = context.get("bicameral_advisory")
        if not isinstance(frame, dict):
            return None
        try:
            from core.brain.bicameral_advisory import get_bicameral_advisory

            learned = get_bicameral_advisory().learn_from_feedback(
                frame,
                reward=float(reward),
                outcome=outcome,
            )
            return learned if isinstance(learned, dict) else None
        except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="continued cognitive cycle without bicameral advisory feedback learning",
            )
            logger.debug("Bicameral advisory feedback learning skipped: %s", exc)
        return None

    async def think(
        self,
        objective: str,
        context: dict[str, Any] = None,
        mode: ThinkingMode = ThinkingMode.FAST,
        origin: str | None = None,
        **kwargs,
    ) -> Thought:
        """Execute a cognitive cycle, under one turn ledger and one finalizer.

        The cycle used to be able to end holding an answer. Every gate below
        this point that rejects a draft records it on the bound turn instead
        of destroying it, so when the cycle would otherwise return an empty
        thought there is somewhere to ask "do we actually have something?".

        The binding is a contextvar, so concurrent turns do not share a
        ledger, and the finalizer runs exactly once per cycle in the
        ``finally`` — including when a phase raises, which is precisely the
        path that used to leave no record of what the turn had.
        """
        # Refuse before binding a turn ledger, not after. think_stream and
        # generate already did this; think — the primary entry point — did
        # not, so stop() left the busiest path running. A stop that stops two
        # of three entries has not stopped anything a caller can rely on.
        self._refuse_if_stopped("think")

        # Restore the antecedent for a message that cannot stand alone. Live
        # on 2026-08-03: "Can you do it now?" after a refused screen read, and
        # "From the grant research funds manager" answering Aura's own
        # "Response from who?" — both were reasoned about as though the
        # conversation had just begun, because every path below receives one
        # message and no turn before it.
        #
        # Done here rather than at each caller because think() is the single
        # chokepoint they all pass through, and a fix applied per-route is a
        # fix that the next route will not have.
        objective = self._objective_with_antecedent(objective)

        # A turn already bound by the caller is THE turn, and this one joins
        # it rather than opening a second.
        #
        # The scope used to end when think() returned, and the reply is not
        # delivered when think() returns — every honesty gate, repair pass and
        # shaping stage in the route runs afterwards. So the ledger that exists
        # to say what happened to a turn closed before the part of the turn
        # where things happen to it, and anything asking `current_turn()` from
        # the delivery path got None. Opening a second outcome there would give
        # two answers to "which turn is this", which is the failure one level
        # up from the one it fixes.
        adopted = current_turn()
        outcome = adopted if adopted is not None else TurnOutcome(origin=str(origin or "unknown"))
        owns_outcome = adopted is None
        turn_id = str(uuid.uuid4())
        try:
            # A fluent reply proves nothing about which architecture produced
            # it. The quick lane, a canonical pre-rendered floor, the full
            # phase pipeline and reactive recovery are indistinguishable
            # downstream, and only one of them is the mind the demo is taken to
            # demonstrate. This records which one ran, phase by phase, so the
            # question has an answer instead of a claim.
            with bind_turn(outcome), recording_turn(
                turn_id,
                phases_available=[phase.__class__.__name__ for phase in self._phases],
            ) as turn_receipt:
                thought = await self._think_within_turn(
                    objective, context, mode, origin, **kwargs
                )
                _attach_turn_receipt(thought, turn_receipt)
        except BaseException as exc:
            outcome.record_error(
                f"{type(exc).__name__}: {exc}",
                retryable=not isinstance(exc, (MemoryError, SystemExit, KeyboardInterrupt)),
            )
            if owns_outcome:
                finalize_turn(outcome, subsystem="cognitive_engine")
            raise

        content = str(getattr(thought, "content", "") or "").strip()
        if content:
            outcome.mark_served(content)
        else:
            # No answer surfaced. Before this is recorded as a failed cycle,
            # ask the ledger whether a gate suppressed something servable —
            # a recoverable draft here IS the live defect, and the finalizer
            # escalates it by name rather than as generic infrastructure noise.
            outcome.mark_served("", state=UserVisibleState.NOTHING_SERVED)
        # An adopted turn is finalized by whoever opened it, after the reply is
        # actually delivered. Finalizing someone else's ledger here would close
        # it before the delivery stages that the ledger exists to observe.
        if owns_outcome:
            finalize_turn(outcome, subsystem="cognitive_engine")
        return thought

    async def _think_within_turn(
        self,
        objective: str,
        context: dict[str, Any] = None,
        mode: ThinkingMode = ThinkingMode.FAST,
        origin: str | None = None,
        **kwargs,
    ) -> Thought:
        """
        Execute a cognitive cycle to produce a thought.
        This now drives the 8 phases to transform state.
        """
        self._refuse_if_stopped("think")
        origin = self._resolve_origin(origin, context)
        context = context if isinstance(context, dict) else {}
        mode = self._normalize_mode(mode)
        is_background = self._is_background_request(
            origin, bool(kwargs.get("is_background", False))
        )

        from core.runtime.cognitive_execution_scope import (
            bind_cognitive_execution_scope,
            resolve_cognitive_execution_scope,
        )

        execution_scope = resolve_cognitive_execution_scope(
            origin=origin,
            context=context,
        )
        context["cognitive_execution_scope"] = execution_scope.value

        if is_background:
            suppression_reason = self._background_suppression_reason()
            if suppression_reason:
                logger.debug(
                    "🛡️ CognitiveEngine: Suppressing background thought for origin=%s (%s).",
                    origin,
                    suppression_reason,
                )
                return self._empty_thought(
                    mode, f"background_thought_suppressed:{suppression_reason}"
                )

        if self._should_suppress_background_reflection(mode, is_background):
            logger.debug(
                "🛡️ CognitiveEngine: Suppressing background %s thought during active service window.",
                mode.name,
            )
            return self._empty_thought(mode, "background_reflection_suppressed")

        # The first 50 characters of every objective went to the log sink as
        # typed. A person's message can open with an address, a token, or a
        # phone number, and log files are read, shipped and kept. The scrubber
        # that already protects the cloud path protects this one too; her
        # working memory keeps the real words, because that is her memory of
        # what was said and this is a diagnostic.
        logger.info(
            "🧠 CognitiveEngine.think: %s... (%s) Origin: %s",
            self._log_safe_objective(objective),
            mode.name,
            origin,
        )

        # 1. Get current state (BUG-12 Fix: handle None state on first boot)
        is_test_run = self._is_test_run(origin)
        if is_test_run:
            from core.state.aura_state import AuraState
            state = AuraState.default()
            logger.info("🧠 CognitiveEngine.think: Enforced database-independent state isolation for test run.")
            if self.state_repository is None:
                container = get_container()
                self.state_repository = container.get("state_repository", default=None)
        else:
            repo = self.state_repository
            if repo is None:
                container = get_container()
                repo = container.get("state_repository", default=None)
                self.state_repository = repo

            if repo is None:
                from core.state.aura_state import AuraState

                state = AuraState.default()
            else:
                state = await repo.get_current()

            if state is None:
                from core.state.aura_state import AuraState

                state = AuraState.default()

        # 2. Derive base state for this cognitive cycle (Zenith-HF12 Fix)
        # This ensures every cycle starts with a unique version to prevent Atomic Guard rejections.
        state = state.derive(f"cognitive_intent: {origin}", origin=origin)

        # 3. Hardening: Set Current Objective & Origin
        # This prevents the race condition where ResponseGeneration would pick up
        # a background motivation message instead of the user's input.
        state.cognition.current_objective = objective
        state.cognition.current_origin = origin
        bind_cognitive_execution_scope(
            state,
            objective,
            execution_scope,
            source=f"cognitive_engine:{origin}",
        )
        _record_objective_binding(
            state,
            objective,
            source=f"cognitive_engine:{origin}",
            mode=mode,
            reason="cognitive_cycle_bound",
        )
        state.response_modifiers["model_tier"] = "tertiary" if is_background else "primary"
        state.response_modifiers["deep_handoff"] = False

        # A promoted, grammar-qualified recurrent program is already a complete
        # cognitive result.  It must own the turn before model-backed advisors,
        # augmentors, or the ordinary response phases spend the resident lane
        # and create competing drafts.  The result still enters the ordinary
        # thinking loop as a direct thought so user memory, durable state,
        # foreground closure, the turn ledger, and delivery all use the same
        # machinery as every other accepted response.
        qualified_reply = await self._qualified_recurrent_direct_reply(
            state,
            objective,
            mode,
            origin,
            context,
            is_background=is_background,
            timeout_s=kwargs.get("timeout_s", kwargs.get("timeout")),
        )
        if qualified_reply is not None:
            loop_kwargs = dict(kwargs)
            loop_kwargs["is_background"] = is_background
            loop_kwargs["precomputed_direct_reply"] = qualified_reply
            return await self._run_thinking_loop(
                state,
                objective,
                mode,
                origin,
                context,
                **loop_kwargs,
            )

        context = self._apply_spiking_active_inference(
            state,
            objective,
            origin,
            context,
            is_background=is_background,
        )
        context = self._apply_imagination_workspace(
            state,
            objective,
            origin,
            context,
            is_background=is_background,
        )
        context = self._apply_entity_memory(state, objective, context)
        context = self._apply_bicameral_advisory(
            state,
            objective,
            origin,
            context,
            is_background=is_background,
        )
        context = self._apply_cognitive_situation_frame(
            state,
            objective,
            origin,
            context,
            is_background=is_background,
        )

        structured = self._structured_evaluation_thought(
            objective,
            state=state,
            mode=mode,
            origin=origin,
            fast_path=is_test_run or origin in {"proof", "eval", "evaluation", "benchmark"},
            context=context,
        )
        if structured is not None:
            return structured

        # v40: Spiritual Spine - Prior Position Injection
        # The ordering is critical: injection -> system prompt -> user message.
        spine = get_container().get("spine", default=None)
        if spine and origin in ("user", "voice", "admin"):
            # Extract topic: look for nouns or use the first sentence.
            # v40: Improved topic extraction
            import re

            # Extract first sentence, then remove common filler
            raw = re.split(r"[.?!]", objective)[0].strip()
            # Remove "Tell me about", "What is", etc.
            topic = re.sub(
                r"(?i)^(tell me about|what is|what are|do you think about|give me|how does)\s+",
                "",
                raw,
            )
            topic = topic[:60] if topic else "general"

            check = await spine.pre_response_check(objective, topic=topic)
            if check.injection:
                logger.info("⚡ [Spine] Binding prior position as system context.")
                # As a SYSTEM message, not spliced into the objective.
                #
                # The injection used to be prepended to `objective`, and the
                # prompt builder appends the whole objective as role=user — so
                # Aura's own prior position was persisted, and re-read on every
                # later turn, as something the PERSON had said. It still
                # influences the cycle (working memory is history, and history
                # reaches the prompt); it is just attributed to the side that
                # produced it.
                state.cognition.working_memory.append(
                    {
                        "role": "system",
                        "content": str(check.injection),
                        "timestamp": time.time(),
                        "metadata": {"type": "spine_prior_position", "topic": topic},
                    }
                )
                state.cognition.modifiers["spine_prior_position"] = str(check.injection)
                _record_objective_binding(
                    state,
                    objective,
                    source=f"cognitive_engine:{origin}",
                    mode=mode,
                    reason="spine_injection_bound",
                )

        # Identity drift: measured, never spliced into the objective.
        #
        # Two injections used to live here and neither could work. The first
        # prepended a correction string ("[SPINE CHECK] Am I agreeing under
        # social pressure?") produced by the drift monitor — asking a
        # drifting process to talk itself out of drifting, and a direct
        # violation of the rule that fixes are causal, not verbal.
        #
        # The second prepended "[IDENTITY REFRESH: REMEMBER WHO YOU ARE]"
        # when the identity anchor looked like a small fraction of the
        # window. It was measuring a RATIO, not an absence:
        # build_system_prompt injects AURA_IDENTITY in full on every turn
        # regardless of depth, so the anchor was never actually missing and
        # the shout added nothing except an instruction to perform identity.
        #
        # Context health is still worth knowing, so it is still computed and
        # logged. It just no longer edits what Aura was asked to do.
        drift = get_container().get("drift_monitor", default=None)
        if drift and background_policy.is_user_facing_origin(origin):
            try:
                hist_len = len(str(state.cognition.working_memory))
                sys_len = len(ContextAssembler.build_system_prompt(state))
                if drift.needs_context_refresh(hist_len, sys_len):
                    logger.info(
                        "[Drift] identity anchor is %.1f%% of the window at depth; "
                        "anchor is still injected in full",
                        (sys_len / hist_len * 100) if hist_len else 100.0,
                    )
            except (AttributeError, TypeError, ValueError, ZeroDivisionError) as _drift_exc:
                record_degradation(
                    "cognitive_engine.drift_context_health",
                    _drift_exc,
                    action="skipped context-health measurement for this turn",
                )

        # v5.2: Augmentor Context Injection
        # Pull signals from registered augmentors before the phase loop
        augmentor_context = {}
        for aug in self._augmentors:
            try:
                if hasattr(aug, "get_augmentation"):
                    # On a thread with a deadline: get_augmentation is
                    # synchronous and ran on the event loop, so one slow
                    # augmentor held every turn in the process behind it.
                    aug_data = await asyncio.wait_for(
                        asyncio.to_thread(aug.get_augmentation, objective),
                        timeout=self._AUGMENTATION_TIMEOUT_S,
                    )
                    if aug_data:
                        augmentor_context[type(aug).__name__] = (
                            self._bounded_augmentation(aug_data)
                        )
            except TimeoutError as e:
                record_degradation(
                    "cognitive_engine",
                    e,
                    severity="warning",
                    action="skipped an augmentor that exceeded its turn budget",
                )
                logger.warning(
                    "Augmentor %s exceeded %.1fs and was skipped.",
                    type(aug).__name__,
                    self._AUGMENTATION_TIMEOUT_S,
                )
            except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as e:
                record_degradation(
                    "cognitive_engine",
                    e,
                    severity="warning",
                    action="skipped failed augmentor and continued cognitive loop",
                )
                logger.warning("Augmentor %s failed: %s", type(aug).__name__, e)

        if augmentor_context:
            context = context or {}
            context.update({"augmentations": augmentor_context})

        loop_kwargs = dict(kwargs)
        loop_kwargs["is_background"] = is_background

        thought = await self._run_thinking_loop(
            state,
            objective,
            mode,
            origin,
            context,
            **loop_kwargs,
        )

        return thought

    async def _qualified_recurrent_direct_reply(
        self,
        state: AuraState,
        objective: str,
        mode: ThinkingMode,
        origin: str,
        context: dict[str, Any],
        *,
        is_background: bool,
        timeout_s: Any,
    ) -> Thought | None:
        """Return one certified recurrent answer before general generation."""

        if is_background or not self._is_user_facing_origin(origin):
            return None
        if str(origin or "").strip().lower() in {
            "proof",
            "eval",
            "evaluation",
            "benchmark",
        }:
            return None
        if bool(
            context.get("proof_or_benchmark")
            or context.get("proof_run")
            or context.get("benchmark_run")
        ):
            return None

        from core.brain.llm.qualified_recurrent_ingress import (
            admit_qualified_recurrent_objective,
        )
        from core.conversation.user_surface_contract import (
            bind_user_surface_prompt,
            resolve_user_surface_prompt,
        )

        surface = resolve_user_surface_prompt(context, fallback=objective)
        if surface.bound and not surface.valid:
            record_degradation(
                "cognitive_engine.qualified_recurrent_surface",
                RuntimeError(surface.error or "user_surface_prompt_invalid"),
                severity="warning",
                action="continued through ordinary cognition after rejecting an invalid user-surface binding",
                enforce_failure_policy=False,
            )
            return None
        if not surface.bound:
            bind_user_surface_prompt(
                context,
                surface.prompt or objective,
                source="cognitive_engine.qualified_recurrent_ingress",
                overwrite=True,
            )
            surface = resolve_user_surface_prompt(context, fallback=objective)
        visible_objective = str(surface.prompt or "").strip()
        if not visible_objective:
            return None

        # Admission is answer-blind and total over its supported public
        # grammars.  Checking it here prevents unsupported conversation from
        # touching the latent service or acquiring any model resource.
        try:
            admission = admit_qualified_recurrent_objective(visible_objective)
        except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "cognitive_engine.qualified_recurrent_admission",
                exc,
                severity="warning",
                action="continued through ordinary cognition after typed admission failed",
                enforce_failure_policy=False,
            )
            return None
        if admission is None:
            return None

        try:
            requested_timeout = float(timeout_s) if timeout_s is not None else 8.0
        except (TypeError, ValueError, OverflowError):
            requested_timeout = 8.0
        qualified_timeout = max(1.0, min(8.0, requested_timeout))

        from core.brain.foreground_latent_runtime import (
            run_foreground_latent_episode,
        )

        outcome = await run_foreground_latent_episode(
            orchestrator=None,
            messages=[],
            visible_objective=visible_objective,
            foreground=True,
            desktop_required=bool(
                context.get("desktop_cognitive_engine_required")
                or context.get("cognitive_engine_required")
            ),
            cognitive_mode=str(mode.name).lower(),
            request_timeout_s=qualified_timeout,
            strict_output_contract=bool(context.get("strict_output_contract")),
            incompatible_contract=bool(context.get("incompatible_output_contract")),
            proof_or_benchmark=False,
            tenant_id=str(context.get("tenant_id") or "local"),
            user_id=str(context.get("principal_id") or context.get("user_id") or "owner"),
            session_id=str(context.get("session_id") or "local"),
            domain="desktop_conversation",
        )
        trace = dict(outcome.trace or {})
        state.response_modifiers.update(trace)
        if not str(outcome.text or "").strip():
            if trace.get("qualified_recurrent_eligible"):
                reason = str(
                    trace.get("qualified_recurrent_reason")
                    or trace.get("latent_cortex_failure_reason")
                    or "qualified_recurrent_disposition_missing"
                )
                logger.warning(
                    "Qualified recurrent ingress admitted the user request but did not "
                    "produce a serving answer: %s",
                    reason,
                )
            return None

        response_path = "cognitive_engine_qualified_recurrent"
        state.response_modifiers["response_path"] = response_path
        state.response_modifiers["model_tier"] = "certified_recurrent"
        evidence = tuple(str(item) for item in outcome.evidence if str(item))
        logger.info(
            "Qualified recurrent ingress served family=%s path=%s",
            getattr(admission, "family", "unknown"),
            response_path,
        )
        return Thought(
            id=str(uuid.uuid4()),
            content=str(outcome.text).strip(),
            mode=mode,
            confidence=0.95,
            reasoning=["Certified qualified recurrent execution completed."],
            metadata={
                **trace,
                "response_path": response_path,
                "qualified_recurrent_family": str(
                    getattr(admission, "family", "unknown")
                ),
                "qualified_recurrent_evidence": evidence,
                "live_mind_generation_required": False,
                "model_generation_used": False,
            },
        )

    async def _run_thinking_loop(
        self,
        state: AuraState,
        objective: str,
        mode: ThinkingMode,
        origin: str,
        context: dict[str, Any] = None,
        **kwargs,
    ) -> Thought:
        """
        Internal method to execute the core cognitive phase loop.
        Extracted from `think` to allow pre/post-processing in `think`.
        """
        if not isinstance(context, dict):
            context = {}
        foreground_turn_objective = str(objective or "")
        _bind_live_mind_generation_contract(context)
        from core.conversation.user_surface_contract import (
            bind_user_surface_prompt,
            resolve_user_surface_prompt,
        )

        surface_prompt = resolve_user_surface_prompt(context)
        if not surface_prompt.bound:
            bind_user_surface_prompt(
                context,
                surface_prompt.prompt or objective,
                source="cognitive_engine.visible_user_message",
                overwrite=True,
            )

        append_user_message = True
        append_user_message = not bool(
            context.get("suppress_user_memory_append")
            or context.get("suppress_working_memory_user_append")
        )
        self._thinking_loop_surface_prompt(append_user_message, context, objective, origin, state, surface_prompt)

        is_background = bool(kwargs.get("is_background", False))
        explicit_timeout = kwargs.get("timeout_s", kwargs.get("timeout"))
        try:
            cycle_timeout = float(explicit_timeout) if explicit_timeout is not None else 0.0
        except (TypeError, ValueError):
            cycle_timeout = 0.0
        if cycle_timeout <= 0.0:
            if self._is_user_facing_origin(origin):
                cycle_timeout = 180.0
            elif is_background:
                cycle_timeout = 90.0
            else:
                cycle_timeout = 240.0
        cycle_timeout_cap = _DEFAULT_COGNITIVE_CYCLE_MAX_S
        if explicit_timeout is not None and self._is_user_facing_origin(origin):
            cycle_timeout_cap = response_policy.USER_FACING_COMPLETION_DEADLINE_MAX_S
        cycle_timeout = max(8.0, min(cycle_timeout_cap, cycle_timeout))
        cycle_deadline_at = time.monotonic() + cycle_timeout
        context["cognitive_cycle_deadline_monotonic"] = cycle_deadline_at

        # 4. Phase Execution Loop with Watchdog
        import copy

        backup_state = copy.deepcopy(state)
        temp_state = state
        success = False
        # Where this turn's working memory begins. An assistant message at or
        # before this mark belongs to an EARLIER turn: a duplicate or
        # suppressed user append leaves one sitting at the end, and extraction
        # checked only `role == "assistant"`, so the previous answer went out
        # again as this one's.
        turn_memory_mark = len(getattr(state.cognition, "working_memory", []) or [])
        # What this turn changed, so a rebase can restore only that. Restoring
        # everything from the per-turn snapshot overwrote whatever a concurrent
        # writer had done to goals, initiatives, focus and modifiers.
        pre_turn_cognition = {
            "active_goals": list(getattr(state.cognition, "active_goals", []) or []),
            "pending_initiatives": list(
                getattr(state.cognition, "pending_initiatives", []) or []
            ),
            "attention_focus": getattr(state.cognition, "attention_focus", None),
            "phenomenal_state": getattr(state.cognition, "phenomenal_state", None),
            "modifiers": dict(getattr(state.cognition, "modifiers", {}) or {}),
        }

        direct_quick_reply = kwargs.pop("precomputed_direct_reply", None)
        if direct_quick_reply is None:
            direct_quick_reply = await self._direct_desktop_quick_reply(
                objective,
                mode,
                origin,
                context,
                timeout_s=cycle_timeout,
            )
        if direct_quick_reply is not None:
            # The quick lane returned before any phase executed. Whether the
            # model was called depends on which branch inside it answered: the
            # canonical floors return pre-rendered text and never reach it.
            record_response_path(
                str(
                    (direct_quick_reply.metadata or {}).get("response_path")
                    or "desktop_quick_reply"
                ),
                model_generation=bool(
                    (direct_quick_reply.metadata or {}).get(
                        "live_mind_generation_required", True
                    )
                ),
            )
            state.cognition.working_memory.append(
                {
                    "role": "assistant",
                    "content": direct_quick_reply.content,
                    "timestamp": time.time(),
                    "origin": origin,
                }
            )
            if self._is_user_facing_origin(origin):
                state.transition_origin = origin
                state.cognition.current_origin = origin
            temp_state = state
            success = True

        if not success:
            # Bound before the try, because the finally below reads it and the
            # timeout context manager can fail on entry.
            _provenance_tick = None
            try:
                async with asyncio.timeout(cycle_timeout):
                    _begin_pass_run("legacy_pipeline")
                    # The provenance graph had the same asymmetry the pass
                    # instrumentation had, for the same reason: it was opened
                    # in AuraKernel.tick, and chat drives THIS loop. So the
                    # causal record that answers "why did she do that" existed
                    # for the three turns a day the kernel runs and not for the
                    # several hundred a person has. Same seam, same graph.
                    _provenance_tick = _open_provenance_tick(
                        objective=objective, priority=self._is_user_facing_origin(origin)
                    )
                    for phase in self._phases:
                        phase_name = phase.__class__.__name__
                        # [PASS INSTRUMENTATION] AURA_PASS_BISECT_LIMIT and
                        # AURA_PASS_TRACE existed only in pass_manager and the
                        # kernel tick — and chat drives THIS loop, not the
                        # kernel, by roughly 479 turns to 3. The documented way
                        # to find which phase ruined an answer therefore did
                        # nothing on the path that produces almost every
                        # answer. Same seam, same flag, same ordinals.
                        run_it, ordinal, reason = _pass_instrumentation().should_run(
                            f"legacy_pipeline/{phase_name}"
                        )
                        if not run_it:
                            _record_legacy_pass(
                                phase_name, ordinal, 0.0, skipped=True, reason=reason
                            )
                            # A skipped phase is part of the causal record. A
                            # graph that shows only what ran cannot answer why
                            # something did NOT happen, which is half of what
                            # "why did you do that" usually means.
                            _skip_provenance(phase_name, temp_state, reason)
                            continue
                        started_at = time.perf_counter()
                        # Measured around the phase, never reported by it.
                        _transformation = _begin_provenance(phase_name, temp_state)
                        _phase_error = ""
                        try:
                            # Pass through kwargs like is_background if phases support it
                            temp_state = await phase.execute(
                                temp_state,
                                objective=objective,
                                context=context,
                                **kwargs,
                            )
                        except BaseException as phase_exc:
                            _phase_error = f"{type(phase_exc).__name__}: {phase_exc}"
                            _record_legacy_pass(
                                phase_name,
                                ordinal,
                                time.perf_counter() - started_at,
                                skipped=False,
                                error=_phase_error,
                            )
                            raise
                        finally:
                            _complete_provenance(
                                _transformation,
                                temp_state,
                                error=_phase_error,
                                objective=objective,
                            )
                        _record_legacy_pass(
                            phase_name,
                            ordinal,
                            time.perf_counter() - started_at,
                            skipped=False,
                        )
                        # Marked after the phase returns, so a phase that timed
                        # out mid-execution is not recorded as having run.
                        record_phase(phase_name)

                    state = temp_state
                    record_response_path(
                        "full_phase_pipeline", model_generation=True
                    )
                    if self._is_user_facing_origin(origin):
                        state.transition_origin = origin
                        final_origin = getattr(state.cognition, "current_origin", "")
                        if is_foreground_objective_origin(final_origin) or not str(
                            final_origin or ""
                        ).strip():
                            state.cognition.current_origin = origin
                    success = True
            except TimeoutError:
                logger.error("🛑 [COGNITION] Watchdog: Cognitive cycle TIMEOUT (%.1fs).", cycle_timeout)
                record_response_path("reactive_recovery_timeout", model_generation=False)
                # Immediate Reactive Recovery
                return await self._reactive_recovery(
                    objective,
                    mode,
                    origin,
                    "timeout",
                    context=context,
                    # The version THIS turn derived. A rollback that cannot
                    # match it is undoing somebody else's work.
                    authored_version=int(getattr(state, "version", 0) or 0),
                )
            except (sqlite3.Error, *_COGNITIVE_ENGINE_RECOVERABLE_ERRORS) as e:
                # This caught sqlite3.Error and OSError only, so the failures a
                # phase ACTUALLY produces — RuntimeError, AttributeError,
                # TypeError, ValueError from a malformed return or an
                # implementation defect — escaped the cognitive API entirely.
                # The caller got a raw exception where reactive recovery was
                # the designed behaviour, and the degradation record that names
                # the phase was never written.
                record_degradation(
                    "cognitive_engine",
                    e,
                    severity="critical",
                    action="downshifted or entered reactive recovery after phase failure",
                )
                logger.error("🚨 [COGNITION] Fatal error in phase logic: %s", e)
                # v14.1 HARDENING: Rollback & Downshift
                if mode == ThinkingMode.DEEP:
                    logger.warning(
                        "🔄 [COGNITION] Downshifting to REACTIVE mode due to Deep Failure..."
                    )
                    # WITH the context. The downshift used to call think()
                    # without it, so the retry lost the desktop-required
                    # flags, the live-mind evidence, the scoped request
                    # metadata and the recent exchanges — and then answered a
                    # question it could no longer see the terms of. A retry
                    # that drops the contract is a different request.
                    return await self.think(
                        objective,
                        context=dict(context or {}),
                        mode=ThinkingMode.FAST,
                        origin=origin,
                        **kwargs,
                    )

                record_response_path("reactive_recovery_crash", model_generation=False)
                return await self._reactive_recovery(
                    objective,
                    mode,
                    origin,
                    f"crash: {e}",
                    context=context,
                    authored_version=int(getattr(state, "version", 0) or 0),
                )
            finally:
                # Closed here rather than after the loop so a tick that timed
                # out or crashed still lands in the ring. Those are the ticks
                # somebody most wants to read afterwards, and the version that
                # closed on the success path recorded only the turns that went
                # well.
                _close_provenance_tick(_provenance_tick)
                try:
                    # vResilience: Avoid locals().get() for type stability
                    if not success and "backup_state" in locals():
                        # This restores a LOCAL REFERENCE and nothing else.
                        #
                        # A deep copy of the state object cannot undo what a
                        # phase already did outside it: events published, tools
                        # invoked, rows written, in-place mutations to
                        # collaborators the phase was handed. Calling this
                        # "rollback" invites the next reader to rely on a
                        # transaction that does not exist, so the receipt says
                        # what was and was not restored.
                        state = backup_state
                        self._last_phase_rollback = {
                            "restored": "cognitive_state_snapshot",
                            "not_restored": [
                                "external_service_writes",
                                "published_events",
                                "tool_invocations",
                                "database_rows",
                                "in_place_collaborator_mutations",
                            ],
                            "at": time.time(),
                        }
                        record_degradation(
                            "cognitive_engine",
                            RuntimeError("phase_failure_partial_rollback"),
                            severity="warning",
                            action="restored the cognitive state snapshot; external phase effects are not reversible here",
                        )
                except (OSError, ConnectionError, TimeoutError) as _e:
                    record_degradation(
                        "cognitive_engine",
                        _e,
                        severity="warning",
                        action="continued with current state after backup restore check failed",
                    )
                    logger.debug("Ignored Exception in cognitive_engine.py: %s", _e)

        # Capture the routed objective before closing a foreground turn. Response
        # extraction still needs it for action-imperative validation, but durable
        # state must not retain a completed chat turn as autonomous work.
        routed_obj = str(getattr(state.cognition, "current_objective", "") or "")
        is_action_imperative = (
            "[ACTION IMPERATIVE]" in objective or "[ACTION IMPERATIVE]" in routed_obj
        )
        # finalize_foreground_turn_state mutates the state about to be
        # committed, so it belongs here. The CLOSURE notification does not: it
        # tells external lifecycle state that the turn completed, and it used
        # to fire before persistence, so a bypassed or failed commit left the
        # rest of the runtime believing a turn had completed that durable state
        # has no record of. It moves below the commit loop.
        _is_foreground_turn = self._is_user_facing_origin(origin) and not is_background
        if _is_foreground_turn:
            finalize_foreground_turn_state(
                state,
                objective=foreground_turn_objective,
                origin=origin,
            )

        # ─── SUCCESS PATH (Unreachable before fix) ──────────────────────────
        # 5. Final State Commit
        # HF12: Handle concurrent version conflicts with a mini-retry loop
        is_test_run = self._is_test_run(origin)
        should_bypass_commit = is_test_run or self.state_repository is None
        # The watchdog above wraps PHASE EXECUTION only. Repository reads,
        # advisors, spine checks, augmentors, the deep copy, this commit loop
        # and feedback learning all run outside it, so a configured cycle
        # timeout was never the end-to-end budget it reads as. The commit loop
        # is the largest of those — three attempts, each a database round trip
        # — so it gets what is left of the same deadline instead of an
        # unbounded wait after the budget is already gone.


        # What actually happened to durable state. Every exit from this loop
        # used to be a bare `break`, after which extraction returned a
        # 0.9-confidence "completed successfully" thought whether the commit
        # had landed, been bypassed, exhausted its retries, or raised.
        commit_outcome = "not_attempted"
        max_retries = 3
        commit_outcome, state = await _commit_the_thought_with_retries(
            commit_outcome=commit_outcome,
            cycle_deadline_at=cycle_deadline_at,
            is_test_run=is_test_run,
            max_retries=max_retries,
            origin=origin,
            pre_turn_cognition=pre_turn_cognition,
            self=self,
            should_bypass_commit=should_bypass_commit,
            state=state,
            temp_state=temp_state,
        )

        # The turn completed durably (or was legitimately isolated). Only now
        # may external lifecycle state be told it finished.
        if _is_foreground_turn and commit_outcome in {
            "committed",
            "bypassed_test_isolation",
        }:
            closure = get_container().get("executive_closure", default=None)
            if closure is not None and hasattr(closure, "complete_foreground_turn"):
                closure.complete_foreground_turn(foreground_turn_objective, origin)
        elif _is_foreground_turn:
            record_degradation(
                "cognitive_engine",
                RuntimeError(f"foreground_turn_uncommitted:{commit_outcome}"),
                severity="warning",
                action="withheld foreground closure because cognitive state did not commit",
            )

        # 6. Extract Response
        last_msg = self._turn_response_message(
            state.cognition.working_memory, mark=turn_memory_mark
        )
        if last_msg:
            self.autopoiesis.experience_friction(objective[:20], 0.05)
            # Reward is no longer 1.0 for the mere existence of an
            # assistant-shaped message. Three learning systems were given the
            # maximum positive signal with no user feedback, no correctness
            # check, no tool postcondition and no persistence outcome — an
            # answer that failed to commit taught them it had gone perfectly.
            _cycle_reward = 1.0 if commit_outcome in {
                "committed",
                "bypassed_test_isolation",
            } else 0.5
            feedback = self._learn_spiking_active_inference_outcome(
                context,
                outcome="assistant_response",
                reward=_cycle_reward,
            )
            if direct_quick_reply is not None:
                thought = direct_quick_reply
                quick_metadata = dict(thought.metadata or {})
                imagination_feedback = quick_metadata.get(
                    "imagination_workspace_feedback"
                )
                if not isinstance(imagination_feedback, dict):
                    imagination_feedback = self._learn_imagination_workspace_outcome(
                        context,
                        outcome="assistant_response",
                        reward=_cycle_reward,
                    )
                bicameral_feedback = quick_metadata.get("bicameral_advisory_feedback")
                if not isinstance(bicameral_feedback, dict):
                    bicameral_feedback = self._learn_bicameral_advisory_outcome(
                        context,
                        outcome="assistant_response",
                        reward=_cycle_reward,
                    )
                thought.metadata = {
                    **quick_metadata,
                    "spiking_active_inference_feedback": feedback,
                    "imagination_workspace_feedback": imagination_feedback,
                    "bicameral_advisory_feedback": bicameral_feedback,
                }
            else:
                imagination_feedback = self._learn_imagination_workspace_outcome(
                    context,
                    outcome="assistant_response",
                    reward=_cycle_reward,
                )
                bicameral_feedback = self._learn_bicameral_advisory_outcome(
                    context,
                    outcome="assistant_response",
                    reward=_cycle_reward,
                )
                generation_controls = context.get("live_mind_generation_controls")
                if not isinstance(generation_controls, dict):
                    generation_controls = {}
                surface_control_receipt = state.response_modifiers.get(
                    "live_mind_surface_control_receipt"
                )
                if not isinstance(surface_control_receipt, dict):
                    surface_control_receipt = {}
                if not surface_control_receipt:
                    try:
                        router = get_container().get("llm_router", default=None)
                        if router is not None and hasattr(
                            router, "get_last_generation_metadata"
                        ):
                            generation_metadata = router.get_last_generation_metadata()
                            if isinstance(generation_metadata, dict):
                                candidate = generation_metadata.get(
                                    "surface_control_receipt"
                                )
                                if isinstance(candidate, dict):
                                    surface_control_receipt = dict(candidate)
                    except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
                        logger.debug(
                            "Could not read full-phase surface-control receipt: %s",
                            exc,
                        )
                context_controls_bound = bool(
                    context.get("live_mind_controls_bound", False)
                    and generation_controls
                )
                surface_control_receipt = normalize_live_mind_surface_control_receipt(
                    surface_control_receipt,
                    controls_bound=context_controls_bound,
                    generation_controls=generation_controls,
                    source="cognitive_engine_full_phase_controls",
                )
                latent_final_quality = state.response_modifiers.get(
                    "latent_cortex_final_output_quality"
                )
                latent_quality_reasons = (
                    tuple(latent_final_quality.get("reasons") or ())
                    if isinstance(latent_final_quality, dict)
                    else ()
                )
                surface_reasons = tuple(
                    dict.fromkeys(
                        (
                            *tuple(
                                surface_control_receipt.get(
                                    "surface_quality_gate_reasons"
                                )
                                or ()
                            ),
                            *latent_quality_reasons,
                        )
                    )
                )
                generation_stop_reason = str(
                    surface_control_receipt.get("generation_stop_reason") or ""
                )
                semantic_completion_incomplete = bool(
                    surface_control_receipt.get("semantic_completion_incomplete", False)
                )
                full_phase_text = str(last_msg.get("content") or "").strip()
                generation_failure_class = str(
                    state.response_modifiers.get("generation_failure_class") or ""
                ).lower()
                reply_generation_incomplete = bool(
                    semantic_completion_incomplete
                    or set(surface_reasons)
                    & {
                        "truncated_tail",
                        "final_answer_missing",
                        "missing_final_answer",
                        "incomplete_code_response",
                    }
                    or generation_stop_reason
                    in {"max_tokens", "deadline_exceeded", "soft_cancelled"}
                    or any(
                        reason in generation_failure_class
                        for reason in (
                            "truncated_tail",
                            "final_answer_missing",
                            "missing_final_answer",
                            "incomplete_code_response",
                        )
                    )
                    or _truncation_verdict(
                        full_phase_text,
                        generation_stop_reason=generation_stop_reason,
                    )
                )
                latent_metadata = {
                    key: state.response_modifiers.get(key)
                    for key in (
                        "latent_cortex_selected",
                        "latent_cortex_selection_reason",
                        "latent_cortex_depth_worthy",
                        "latent_cortex_prompt_shape",
                        "latent_cortex_attempted",
                        "latent_cortex_succeeded",
                        "latent_cortex_fallback_used",
                        "latent_cortex_failure_reason",
                        "latent_cortex_identity_bound",
                        "latent_cortex_final_text_transformed",
                        "latent_cortex_final_output_quality",
                        "latent_cortex_raw_final_quality_hash_match",
                        "latent_cortex_receipt",
                        "latent_cortex_ingress",
                        "latent_cortex_progress",
                    )
                    if key in state.response_modifiers
                }
                _degraded_count = len(
                    [
                        key
                        for key in state.response_modifiers
                        if str(key).endswith("_degraded")
                        and state.response_modifiers.get(key)
                    ]
                )
                thought = Thought(
                    id=str(uuid.uuid4()),
                    content=last_msg["content"],
                    mode=mode,
                    confidence=self._cycle_confidence(
                        commit_outcome=commit_outcome,
                        degraded_subsystems=_degraded_count,
                    ),
                    reasoning=[
                        "Phase-based cognitive cycle completed.",
                        f"State commit: {commit_outcome}.",
                    ],
                    metadata={
                        "state_commit_outcome": commit_outcome,
                        "spiking_active_inference": context.get("spiking_active_inference")
                        if isinstance(context, dict)
                        else None,
                        "spiking_active_inference_feedback": feedback,
                        "imagination_workspace_feedback": imagination_feedback,
                        "bicameral_advisory": context.get("bicameral_advisory")
                        if isinstance(context, dict)
                        else None,
                        "bicameral_advisory_feedback": bicameral_feedback,
                        "cognitive_situation_frame": context.get("cognitive_situation_frame")
                        if isinstance(context, dict)
                        else None,
                        "live_mind_controls_bound": context_controls_bound,
                        "live_mind_generation_controls": dict(generation_controls),
                        "live_mind_snapshot_ready": bool(
                            context.get("live_mind_snapshot_ready", False)
                        ),
                        "live_mind_required_subsystems_ok": bool(
                            context.get("live_mind_required_subsystems_ok", False)
                        ),
                        "live_mind_context_required": bool(
                            context.get("live_mind_context_required", False)
                        ),
                        "live_mind_surface_control_receipt": dict(
                            surface_control_receipt
                        ),
                        "live_mind_controls_worker_applied": bool(
                            surface_control_receipt.get("live_mind_controls_bound")
                            and surface_control_receipt.get("applied")
                        ),
                        "reply_generation_incomplete": reply_generation_incomplete,
                        "reply_generation_stop_reason": generation_stop_reason,
                        "reply_generation_failure_reasons": surface_reasons,
                        "reply_original_chars": len(full_phase_text),
                        **latent_metadata,
                        "response_path": str(
                            state.response_modifiers.get("response_path")
                            or (
                                "cognitive_engine_latent_cortex"
                                if state.response_modifiers.get(
                                    "latent_cortex_succeeded"
                                )
                                is True
                                else "cognitive_engine"
                            )
                        ),
                    },
                )
            self.thoughts.append(thought)
            return thought

        # Record the pressure, then READ it back. Until this, the graph had
        # two writers and no reader anywhere in the codebase: it accumulated
        # friction that could not influence any output, which makes the
        # signal unmeasurable rather than merely unused.
        friction_key = objective[:20]
        self.autopoiesis.experience_friction(friction_key, 0.45)
        if self.autopoiesis.is_under_pressure(friction_key):
            logger.warning(
                "Objective '%s' keeps failing to resolve (friction %.2f); "
                "repeated failures on one kind of request are a defect signal, "
                "not noise",
                friction_key,
                self.autopoiesis.friction_for(friction_key),
            )
            record_degradation(
                "cognitive_engine",
                RuntimeError(f"objective repeatedly unresolved: {friction_key}"),
                severity="warning",
                action="recorded sustained objective friction",
                extra=self.autopoiesis.pressure_report(),
                # Friction is a durable learning/diagnostic observation, not
                # an exception from the service contract. Letting the generic
                # fail-closed policy enforce this warning turned a useful
                # signal into CRITICAL SERVICE FAILURE and killed the repair
                # pass that was supposed to resolve it.
                enforce_failure_policy=False,
            )
        self._learn_spiking_active_inference_outcome(
            context,
            outcome="no_assistant_response",
            reward=-0.65,
        )
        self._learn_imagination_workspace_outcome(
            context,
            outcome="no_assistant_response",
            reward=-0.65,
        )
        self._learn_bicameral_advisory_outcome(
            context,
            outcome="no_assistant_response",
            reward=-0.65,
        )

        # ── ACTION IMPERATIVE FALLBACK ──
        #
        # This used to emit [SOMATIC:key='.'] for ANY turn whose objective
        # contained "[ACTION IMPERATIVE]" — text a user can type and injected
        # content can carry — and called it a safe no-op. A keystroke is not a
        # no-op: it goes to whatever holds focus, which may be a terminal, an
        # editor, or a form. The legitimate somatic reflex
        # (orchestrator/mixins/incoming_logic.py) fires only on a message
        # carrying [EMBODIED CONTROL CONTRACT] and only when a real CLI prompt
        # pattern matched. The same condition governs it here: without the
        # contract, a turn that produced no response says so.
        if is_action_imperative:
            embodied_control = "[EMBODIED CONTROL CONTRACT]" in objective or (
                "[EMBODIED CONTROL CONTRACT]" in routed_obj
            )
            if embodied_control:
                logger.warning(
                    "⚠️ [COGNITION] Embodied control turn produced no response. "
                    "Falling back to the pager-advance key."
                )
                return Thought(
                    id=str(uuid.uuid4()),
                    content="[SOMATIC:key='.']",
                    mode=mode,
                    confidence=0.5,
                    reasoning=["Embodied control fallback (pager advance)."],
                    metadata={"embodied_control_contract": True},
                )
            record_degradation(
                "cognitive_engine",
                RuntimeError("action_imperative_without_embodied_contract"),
                severity="warning",
                action="refused a motor fallback for an action imperative with no embodied control contract",
            )
            logger.warning(
                "⚠️ [COGNITION] Action Imperative active but no response generated, "
                "and no embodied control contract authorises a keystroke."
            )
            return self._empty_thought(mode, "action_imperative_no_response")

        if is_background:
            logger.debug(
                "🛡️ CognitiveEngine: background cycle for origin=%s produced no response; returning quiet no-op.",
                origin,
            )
            return self._empty_thought(mode, "background_cycle_no_response")

        structured = self._structured_evaluation_thought(
            objective,
            state=state,
            mode=mode,
            origin=origin,
            fast_path=False,
            context=context,
        )
        if structured is not None:
            return structured

        # If the objective requires a strict answer format, do not return conversational evasive fallbacks.
        # Instead, attempt a direct, single-turn LLM generation as a high-fidelity recovery mechanism.
        # A literal "<answer>" ANYWHERE in the objective used to activate this
        # recovery — text a person can type, and text injected content can
        # carry — and the recovery sends the full objective to a cloud
        # provider. Routing to a third party is not something the prompt gets
        # to decide. The caller's answer_format kwarg is an explicit contract
        # and still counts; a substring in the user's words does not.
        is_strict_answer = "answer_format" in kwargs or bool(
            context.get("strict_answer_contract", False)
        )
        if "<answer>" in objective.lower() and not is_strict_answer:
            logger.info(
                "🛡️ [COGNITION] '<answer>' appears in the objective but no caller "
                "declared a strict-answer contract; not activating cloud recovery."
            )
        if is_strict_answer:
            logger.warning("⚠️ [COGNITION] Structured answer required but phase execution produced no response. Running last-resort direct recovery...")
            try:
                from core.brain.llm_health_router import get_llm_router
                from core.runtime.proof_policy import proof_model_tier
                router = get_llm_router()
                system_prompt = (
                    "You are a precise solver. Solve the user's problem directly. "
                    "Put your final answer strictly inside <answer>...</answer> tags. "
                    "Do not include any conversational preamble."
                )
                recovery_tier = proof_model_tier() if is_test_run else "primary"
                # Last-resort recovery remains on the selected local lane.
                content = await router.think(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": objective}
                    ],
                    origin=f"recovery_{origin}",
                    allow_cloud_fallback=False,
                    prefer_tier=recovery_tier,
                    protected_foreground_lane=recovery_tier == "primary",
                    proof_primary_lane_required=is_test_run and recovery_tier == "primary",
                    proof_evaluation_contract=is_test_run,
                    foreground_request=True,
                )
                # Nonempty was the whole postcondition: the recovery claimed
                # success at confidence 0.8 for any text at all, including
                # text with no <answer> envelope — the one thing the contract
                # promised. A strict answer that does not carry its envelope
                # did not satisfy the contract, and saying it did is what the
                # caller then parses and fails on.
                cleaned = str(content or "").strip()
                envelope_ok = "<answer>" in cleaned.lower() and "</answer>" in cleaned.lower()
                if cleaned and envelope_ok:
                    thought = Thought(
                        id=str(uuid.uuid4()),
                        content=content,
                        mode=mode,
                        confidence=0.8,
                        reasoning=["Last-resort direct structured recovery succeeded."],
                        metadata={"strict_answer_envelope_verified": True},
                    )
                    self.thoughts.append(thought)
                    return thought
                if cleaned:
                    record_degradation(
                        "cognitive_engine",
                        ValueError("strict answer recovery returned no <answer> envelope"),
                        severity="warning",
                        action="refused a strict-answer recovery that did not carry its envelope",
                    )
            except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as rec_err:
                record_degradation(
                    "cognitive_engine",
                    rec_err,
                    severity="degraded",
                    action="returned strict answer recovery failure after direct recovery failed",
                )
                logger.error("Failed last-resort structured recovery: %s", rec_err)
            return self._empty_thought(mode, "strict_answer_recovery_failed")

        # Last stop before this turn tells a person it has nothing. Ask the
        # ledger: did a gate suppress a draft that is still servable? Measured
        # live, the answer was sometimes yes — a complete 240-character reply
        # rejected for `truncated_tail` while the person got an apology.
        #
        # This does NOT overrule the gates. Text they marked unrecoverable
        # (prompt leaks, corrupted output, policy refusals) never comes back;
        # `best_recoverable_candidate` excludes it. What comes back is a draft
        # a heuristic merely disliked, and only when the alternative is
        # nothing at all.
        salvaged = recoverable_answer()
        if not salvaged:
            try:
                from core.conversation.surface_disposition import best_available_reply

                salvaged = best_available_reply(question=objective)
            except (ImportError, RuntimeError, TypeError, ValueError):
                salvaged = ""
        if salvaged:
            generation_metadata = dict(state.response_modifiers)
            generation_controls = context.get("live_mind_generation_controls")
            if not isinstance(generation_controls, dict):
                generation_controls = {}
            surface_control_receipt = generation_metadata.get(
                "live_mind_surface_control_receipt"
            )
            if not isinstance(surface_control_receipt, dict):
                surface_control_receipt = {}
            context_controls_bound = bool(
                context.get("live_mind_controls_bound", False)
                and generation_controls
            )
            surface_control_receipt = normalize_live_mind_surface_control_receipt(
                surface_control_receipt,
                controls_bound=context_controls_bound,
                generation_controls=generation_controls,
                source="cognitive_engine_recoverable_draft_controls",
            )
            latent_final_quality = generation_metadata.get(
                "latent_cortex_final_output_quality"
            )
            latent_quality_reasons = (
                tuple(latent_final_quality.get("reasons") or ())
                if isinstance(latent_final_quality, dict)
                else ()
            )
            surface_reasons = tuple(
                dict.fromkeys(
                    (
                        *tuple(
                            surface_control_receipt.get(
                                "surface_quality_gate_reasons"
                            )
                            or ()
                        ),
                        *latent_quality_reasons,
                    )
                )
            )
            generation_stop_reason = str(
                surface_control_receipt.get("generation_stop_reason") or ""
            )
            generation_failure_class = str(
                generation_metadata.get("generation_failure_class") or ""
            ).lower()
            reply_generation_incomplete = bool(
                surface_control_receipt.get(
                    "semantic_completion_incomplete", False
                )
                or set(surface_reasons)
                & {
                    "truncated_tail",
                    "final_answer_missing",
                    "missing_final_answer",
                    "incomplete_code_response",
                }
                or generation_stop_reason
                in {"max_tokens", "deadline_exceeded", "soft_cancelled"}
                or any(
                    reason in generation_failure_class
                    for reason in (
                        "truncated_tail",
                        "final_answer_missing",
                        "missing_final_answer",
                        "incomplete_code_response",
                    )
                )
                or _truncation_verdict(
                    salvaged,
                    generation_stop_reason=generation_stop_reason,
                )
            )
            recoverable_metadata = {
                key: (dict(value) if isinstance(value, dict) else value)
                for key, value in generation_metadata.items()
                if key
                in {
                    "generation_failure_class",
                    "latent_cortex_selected",
                    "latent_cortex_selection_reason",
                    "latent_cortex_depth_worthy",
                    "latent_cortex_prompt_shape",
                    "latent_cortex_attempted",
                    "latent_cortex_succeeded",
                    "latent_cortex_fallback_used",
                    "latent_cortex_failure_reason",
                    "latent_cortex_identity_bound",
                    "latent_cortex_final_text_transformed",
                    "latent_cortex_final_output_quality",
                    "latent_cortex_raw_final_quality_hash_match",
                    "latent_cortex_receipt",
                    "latent_cortex_ingress",
                    "latent_cortex_progress",
                    "response_path",
                }
            }
            recoverable_metadata.update(
                {
                    "recovered_from_suppression": True,
                    "live_mind_controls_bound": context_controls_bound,
                    "live_mind_generation_controls": dict(generation_controls),
                    "live_mind_snapshot_ready": bool(
                        context.get("live_mind_snapshot_ready", False)
                    ),
                    "live_mind_required_subsystems_ok": bool(
                        context.get("live_mind_required_subsystems_ok", False)
                    ),
                    "live_mind_context_required": bool(
                        context.get("live_mind_context_required", False)
                    ),
                    "live_mind_surface_control_receipt": dict(
                        surface_control_receipt
                    ),
                    "live_mind_controls_worker_applied": bool(
                        surface_control_receipt.get("live_mind_controls_bound")
                        and surface_control_receipt.get("applied")
                    ),
                    "reply_generation_incomplete": reply_generation_incomplete,
                    "reply_generation_stop_reason": generation_stop_reason,
                    "reply_generation_failure_reasons": surface_reasons,
                    "reply_original_chars": len(salvaged),
                }
            )
            logger.warning(
                "🩹 CognitiveEngine: no answer-quality response for origin=%s, but the "
                "turn still held a recoverable %d-char draft; serving it rather than "
                "reporting an empty cycle.",
                origin,
                len(salvaged),
            )
            return Thought(
                id=str(uuid.uuid4()),
                content=salvaged,
                mode=mode,
                confidence=0.4,
                reasoning=["Recovered a gate-suppressed draft; nothing else survived."],
                metadata=recoverable_metadata,
            )

        if bool(
            context.get("desktop_cognitive_engine_required", False)
            or context.get("cognitive_engine_required", False)
        ):
            return self._desktop_cognitive_failure_thought(
                mode,
                str(
                    state.response_modifiers.get("generation_failure_class")
                    or "user_cycle_no_response"
                ),
                generation_metadata=dict(state.response_modifiers),
            )

        logger.warning(
            "🛡️ CognitiveEngine: user-facing cycle for origin=%s produced no answer-quality response.",
            origin,
        )
        return self._empty_thought(mode, "user_cycle_no_response")

    def _thinking_loop_surface_prompt(self, append_user_message, context, objective, origin, state, surface_prompt):
        """Body lifted verbatim out of ``CognitiveEngine._run_thinking_loop``.

        Moved by tools/extract_seam.py, which refuses to write unless the
        relocated body diffs clean against the original. The seam was
        7 names in, 0 out, 0 early return(s), 0 awaits.
        """
        if self._is_user_facing_origin(origin) and append_user_message:
            # WHAT THE PERSON SAID — not what the turn assembled around it.
            #
            # `objective` is the augmented prompt: the visible message plus
            # whatever this turn attached to it — the live-desktop contract
            # directives, grounding evidence, a screen reading, excerpts of
            # her own source. Appending THAT as ``role: user`` records
            # machine-generated instructions as things the person said, and
            # they persist for the rest of the conversation.
            #
            # Measured live 2026-08-04. Two turns about her source code
            # attached real excerpts as evidence; the third turn asked
            # "what's 17 times 4?" and came back with a function from
            # core/memory/associative_entity_memory.py. The excerpts were
            # still in working memory, and text in working memory is
            # material a model continues — the same mechanism that made a
            # screen capture come back as the reply.
            #
            # The visible message is what she should remember being asked.
            from core.utils.injected_blocks import (
                contains_injected_block,
                strip_injected_blocks,
            )

            remembered = strip_injected_blocks(
                str(
                    context.get("visible_user_message")
                    or surface_prompt.prompt
                    or objective
                ).strip()
                or str(objective)
            )

            # A conversation contaminated BEFORE this fix carries those
            # blocks for the rest of its life, and they go on being
            # continued. Scrub what is already there on the way past, so
            # the damage heals instead of persisting.
            for entry in state.cognition.working_memory:
                if not isinstance(entry, dict):
                    continue
                existing = entry.get("content")
                if isinstance(existing, str) and contains_injected_block(existing):
                    entry["content"] = strip_injected_blocks(existing)
            # Check if already in history to avoid duplication
            # vResilience: Workaround for Pyre2 slice limitations
            history = state.cognition.working_memory
            recent_count = min(5, len(history))
            recent = [history[i] for i in range(len(history) - recent_count, len(history))]
            is_duplicate = any(
                m.get("content") in (remembered, objective) for m in recent
            )
            if not is_duplicate:
                # We already derived at the start of the cycle, so we just append here.
                state.cognition.working_memory.append(
                    {
                        "role": "user",
                        "content": remembered,
                        "timestamp": time.time(),
                        "origin": origin,
                    }
                )

    async def _direct_user_facing_recovery(
        self,
        objective: str,
        mode: ThinkingMode,
        origin: str,
        reason: str,
    ) -> Thought | None:
        if not self._is_user_facing_origin(origin):
            return None

        container = get_container()
        router = container.get("llm_router", default=None)
        if router is None or not hasattr(router, "think"):
            return None

        max_tokens = 384 if len(str(objective or "")) <= 900 else 640
        system_prompt = (
            "You are Aura's live CognitiveEngine recovery path. The main phase loop "
            "timed out or failed, but the user still needs one coherent answer. "
            "Answer the current user request directly and honestly. Do not mention "
            "reactive recovery, fallback, internal errors, hidden gates, or implementation "
            "details unless the user specifically asked for them."
        )
        try:
            content = await asyncio.wait_for(
                router.think(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": objective},
                    ],
                    origin=f"recovery_{origin}",
                    prefer_tier="primary",
                    foreground_request=True,
                    protected_foreground_lane=True,
                    is_background=False,
                    deep_handoff=False,
                    allow_deep_handoff=False,
                    allow_cloud_fallback=False,
                    skip_runtime_payload=False,
                    disable_prompt_cache=True,
                    clear_prompt_cache=True,
                    max_tokens=max_tokens,
                    num_predict=max_tokens,
                    timeout=15.0,
                ),
                timeout=17.0,
            )
        except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as rec_err:
            record_degradation(
                "cognitive_engine",
                rec_err,
                severity="degraded",
                action="continued after bounded user-facing direct recovery failed",
            )
            logger.warning("Bounded CognitiveEngine direct recovery failed (%s): %s", reason, rec_err)
            return None

        text = str(content or "").strip()
        if not text or text == "…" or text.startswith("background_thought_suppressed"):
            return None

        thought = Thought(
            id=str(uuid.uuid4()),
            content=text,
            mode=mode,
            confidence=0.65,
            reasoning=[
                f"Bounded user-facing direct recovery succeeded after cognitive failure: {reason}",
                "Recovery used the governed primary router with compact payload and no deep handoff.",
            ],
        )
        self.thoughts.append(thought)
        return thought

    def _desktop_cognitive_failure_thought(
        self,
        mode: ThinkingMode,
        reason: str,
        *,
        generation_metadata: dict[str, Any] | None = None,
    ) -> Thought:
        generation_metadata = (
            dict(generation_metadata)
            if isinstance(generation_metadata, dict)
            else {}
        )
        metadata: dict[str, Any] = {
            "desktop_cognitive_engine_failure": True,
            "failure_reason": str(reason or "unknown")[:240],
            "model_retry_suppressed": True,
        }
        surface_receipt = generation_metadata.get("surface_control_receipt")
        if not isinstance(surface_receipt, dict):
            surface_receipt = generation_metadata.get(
                "live_mind_surface_control_receipt"
            )
        if isinstance(surface_receipt, dict) and surface_receipt:
            metadata["live_mind_surface_control_receipt"] = dict(surface_receipt)
        generation_failure_class = str(
            generation_metadata.get("generation_failure_class")
            or generation_metadata.get("error")
            or reason
            or ""
        ).strip()
        if generation_failure_class:
            metadata["generation_failure_class"] = generation_failure_class[:120]
        for key in (
            "latent_cortex_selected",
            "latent_cortex_selection_reason",
            "latent_cortex_depth_worthy",
            "latent_cortex_prompt_shape",
            "latent_cortex_attempted",
            "latent_cortex_succeeded",
            "latent_cortex_fallback_used",
            "latent_cortex_failure_reason",
            "latent_cortex_identity_bound",
            "latent_cortex_final_text_transformed",
            "latent_cortex_final_output_quality",
            "latent_cortex_raw_final_quality_hash_match",
            "latent_cortex_receipt",
            "latent_cortex_ingress",
            "latent_cortex_progress",
            "live_mind_controls_bound",
            "live_mind_generation_controls",
            "live_mind_snapshot_ready",
            "live_mind_required_subsystems_ok",
            "response_path",
        ):
            if key in generation_metadata:
                value = generation_metadata[key]
                metadata[key] = dict(value) if isinstance(value, dict) else value
        thought = Thought(
            id=str(uuid.uuid4()),
            content=(
                "I couldn't produce a reliable answer to that turn, and I won't "
                "fabricate one. The live Cortex attempt failed its output checks, "
                "so I recorded the failure instead of sending nonsense."
            ),
            mode=ThinkingMode.FAST,
            confidence=0.1,
            reasoning=[f"Desktop CognitiveEngine failure surfaced without model retry: {reason}"],
            metadata=metadata,
        )
        self.thoughts.append(thought)
        return thought

    async def _direct_desktop_quick_reply(
        self,
        objective: str,
        mode: ThinkingMode,
        origin: str,
        context: dict[str, Any] | None,
        *,
        timeout_s: float,
    ) -> Thought | None:
        if not self._is_user_facing_origin(origin):
            return None
        if not isinstance(context, dict) or not bool(context.get("desktop_quick_reply_contract")):
            return None

        container = get_container()
        router = container.get("llm_router", default=None)

        # int() on caller input, outside the guarded router call below: a
        # string or a NaN raised TypeError/ValueError here and took the turn
        # down before any bounded failure thought could be produced. A bad
        # request is a bad request, not a crash.
        max_tokens = self._bounded_request_int(
            context.get("max_tokens"), default=768, low=1, high=32_768
        )
        advice = context.get("spiking_active_inference")
        imagination_frame = context.get("imagination_workspace")
        bicameral_frame = context.get("bicameral_advisory")
        cognitive_situation_frame = context.get("cognitive_situation_frame")
        sampling_sources: list[Any] = []
        if isinstance(advice, dict):
            sampling_sources.append(advice.get("sampling_bias") or {})
        if isinstance(imagination_frame, dict):
            sampling_sources.append(imagination_frame.get("sampling_bias") or {})
        if isinstance(bicameral_frame, dict):
            sampling_sources.append(bicameral_frame.get("sampling_bias") or {})
        if isinstance(cognitive_situation_frame, dict):
            sampling_sources.append(cognitive_situation_frame.get("sampling_bias") or {})
        memory_state_contract = bool(context.get("memory_state_contract", False))
        runtime_fact_status_contract = bool(
            context.get("runtime_fact_status_contract", False)
            or context.get("grounded_runtime_status_contract", False)
        )
        self_condition_contract = bool(context.get("self_condition_contract", False))
        self_condition_contract_covers_turn = bool(
            context.get(
                "self_condition_contract_covers_turn",
                self_condition_contract,
            )
        )
        capability_inventory_contract = bool(context.get("capability_inventory_contract", False))
        identity_continuity_contract = bool(
            context.get("identity_continuity_contract", False)
            or context.get("grounded_identity_continuity_context")
        )
        completion_retry_contract = bool(
            context.get("user_surface_completion_retry", False)
        )
        continuation_partial = continuation_state_text(
            context.get("user_surface_continuation_partial")
        )
        continuation_resume_handle = str(
            context.get("user_surface_continuation_resume_handle") or ""
        ).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{32}", continuation_resume_handle):
            continuation_resume_handle = ""
        continuation_prefix = continuation_prompt_prefix(continuation_partial)
        continuation_contract = bool(
            completion_retry_contract
            and context.get("user_surface_continuation_contract", False)
            and continuation_partial
        )
        obligation_segment = str(
            context.get("user_surface_obligation_segment") or ""
        ).strip()
        obligation_parent_request = str(
            context.get("user_surface_obligation_parent_request") or ""
        ).strip()
        obligation_partial = continuation_state_text(
            context.get("user_surface_obligation_partial")
        )
        obligation_contract = bool(
            completion_retry_contract
            and context.get("user_surface_obligation_contract", False)
            and obligation_segment
            and obligation_parent_request
        )
        prompt_shape = context.get("prompt_shape")
        if not isinstance(prompt_shape, dict):
            prompt_shape = {}
        visible_capacity_request = str(
            context.get("visible_user_message") or objective or ""
        ).strip()
        structural_answer_floor = answer_surface_token_floor(
            visible_capacity_request
        )
        # How much room an answer needs is a property of the question, not of
        # the lane it arrived on. "When does the second train catch the first,
        # and how far from the station?" is a two-part derivation either way;
        # on the quick lane it got the 512 floor, ran out mid-derivation at
        # "The first train has been traveling for 3:00pm + 2.25", and was
        # trimmed back to the last complete sentence — so the answer was never
        # given. Measured live 2026-07-27, along with a recall answer cut at
        # "Probably just read".
        shape_wants_room = bool(
            context.get("bounded_planning_contract", False)
            or prompt_shape.get("prefers_extended_answer", False)
            or prompt_shape.get("requires_single_reply_coverage", False)
            or int(prompt_shape.get("question_parts", 0) or 0) >= 2
            or structural_answer_floor > 256
            or _turn_wants_a_derivation(
                str(context.get("visible_user_message") or objective or "")
            )
        )
        if self_condition_contract_covers_turn:
            # A multi-clause condition check remains one bounded state report.
            # The route has already proved that no planning, execution,
            # retrieval, identity, or memory contract competes for coverage.
            # Treating its evidence distinctions as independent long-form asks
            # previously expanded a 256-token answer into a 1,024-token job.
            shape_wants_room = False
        extended_full_mind_reply = bool(
            context.get("require_full_foreground_mind_reply", False) and shape_wants_room
        )
        canonical_memory_state_evidence = str(
            context.get("canonical_memory_state_evidence") or ""
        ).strip()
        canonical_self_condition_context = str(
            context.get("canonical_self_condition_context") or ""
        ).strip()
        advisory_factors: list[float] = []
        for sampling in sampling_sources:
            if isinstance(sampling, dict):
                try:
                    factor_value = float(sampling.get("max_tokens_factor", 1.0))
                except (TypeError, ValueError):
                    factor_value = 1.0
                if 0.25 <= factor_value <= 1.25:
                    if capability_inventory_contract and factor_value < 1.0:
                        continue
                    advisory_factors.append(factor_value)
        if advisory_factors:
            max_tokens = max(
                128,
                int(max_tokens * _combine_advisory_token_factors(advisory_factors)),
            )
        # A narrow budget requires a narrow turn.
        #
        # These caps are right for what they were written for: "what did I
        # pin?" and "how much RAM?" have bounded, machine-readable answers.
        # A self-condition question is natural conversation: its evidence is
        # structured, but the answer is authored by the resident model and can
        # legitimately need more than a status-line budget. These caps are
        # wrong the moment such a request shares a message with something
        # substantive — the cap then sizes the whole turn by its smallest part.
        # Measured live 2026-07-27:
        # a pin-plus-philosophy message drew 172 tokens, ran out mid-sentence,
        # and the answer was discarded as a truncated tail.
        #
        # memory_state_contract_covers_turn is chat.py reporting what its own
        # parser found; the other two narrow contracts defer to the same
        # question-shape signal the comment above already establishes.
        narrow_state_contract = bool(runtime_fact_status_contract)
        if memory_state_contract and context.get(
            "memory_state_contract_covers_turn", True
        ):
            narrow_state_contract = True
        if narrow_state_contract and not shape_wants_room:
            max_tokens = max(128, min(max_tokens, 256))
        elif capability_inventory_contract and not shape_wants_room:
            max_tokens = max(160, min(max_tokens, 220))
        elif extended_full_mind_reply:
            max_tokens = max(1024, structural_answer_floor, min(max_tokens, 4096))
        elif shape_wants_room:
            # Capacity follows the visible work contract. Natural EOS keeps a
            # short answer short; reducing a five-part answer to a generic
            # middle band only guarantees a later retry.
            max_tokens = max(896, structural_answer_floor, min(max_tokens, 4096))
        else:
            # 512-token floor: a live conversational reply must have room to
            # finish its sentences even after advisory reductions.
            max_tokens = max(512, min(max_tokens, 1024))
        # ``max_tokens`` is already the question-shaped, route-approved budget.
        # Preserve it unless memory pressure is truly critical. The MLX gate
        # keeps 64/32-token critical and emergency caps hard.
        completion_floor = max_tokens
        request_timeout_cap = (
            response_policy.USER_FACING_COMPLETION_DEADLINE_MAX_S
            if shape_wants_room
            else 180.0
        )
        request_timeout = max(
            12.0,
            min(
                max(12.0, float(timeout_s or 32.0) - 5.0),
                request_timeout_cap,
            ),
        )
        if memory_state_contract or runtime_fact_status_contract or self_condition_contract:
            request_timeout = min(request_timeout, 90.0)
        if capability_inventory_contract:
            request_timeout = min(request_timeout, 28.0)
        style_contract = self._contract_safe(
            context.get("response_style_contract"), self._STYLE_CONTRACT_LIMIT
        )
        visible_user_message = str(context.get("visible_user_message") or objective or "").strip()
        recent_conversation_context = str(context.get("recent_conversation_context") or "").strip()
        history_messages = (
            []
            if memory_state_contract or runtime_fact_status_contract
            else _desktop_history_messages_from_context(context)
        )
        if self_condition_contract:
            # Current self-state supersedes old self-descriptions. Preserve
            # prior user context, but do not few-shot the model with assistant
            # answers from earlier samples or rejected drafts.
            history_messages = [
                message
                for message in history_messages
                if str(message.get("role") or "").lower() == "user"
            ]
        live_speech_frame = context.get("live_speech_grounding_frame")
        live_mind_context = context.get("live_mind_context")
        live_mind_required = bool(context.get("live_mind_context_required", False))
        live_mind_generation_controls = _live_mind_generation_controls(
            live_mind_context,
            user_message=visible_user_message,
        )
        if not live_mind_generation_controls and isinstance(
            context.get("live_mind_generation_controls"), dict
        ):
            live_mind_generation_controls = dict(context["live_mind_generation_controls"])
        # The spiking model's temperature and top-p deltas reach the sampler
        # here. Before this they were computed every turn and dropped, leaving
        # a prompt sentence as the neurodynamics' only actuator.
        live_mind_generation_controls = _apply_neurodynamic_sampling_bias(
            live_mind_generation_controls, advice
        )
        live_mind_controls_bound = _live_mind_controls_bound(
            live_mind_context,
            live_mind_generation_controls,
        )
        # The three flags below gate the structured floors, which return
        # self-condition, planning, capability and identity answers at high
        # confidence with live-mind metadata attached. They used to be
        # satisfiable from the caller's own context booleans — and the last
        # branch re-derived controls_bound as True from them, bypassing
        # _live_mind_controls_bound entirely. A caller could therefore mint a
        # proof-bearing reply by asserting that it was entitled to one.
        #
        # A context fallback is still allowed, but only when the payload
        # carries this runtime's stamp: then the booleans are the runtime's own
        # summary of a snapshot it produced, not a claim about itself.
        from core.utils.injected_blocks import is_stamped_runtime_payload

        _context_attested = is_stamped_runtime_payload(live_mind_context)
        live_mind_snapshot_ready = bool(
            isinstance(live_mind_context, dict)
            and isinstance(live_mind_context.get("mind_snapshot_quality"), dict)
            and live_mind_context["mind_snapshot_quality"].get("ready")
        )
        if not live_mind_snapshot_ready and _context_attested:
            live_mind_snapshot_ready = bool(context.get("live_mind_snapshot_ready"))
        live_mind_required_subsystems_ok = bool(
            isinstance(live_mind_context, dict)
            and live_mind_context.get("required_subsystems_ok")
        )
        if not live_mind_required_subsystems_ok and _context_attested:
            live_mind_required_subsystems_ok = bool(
                context.get("live_mind_required_subsystems_ok")
            )
        # controls_bound comes from _live_mind_controls_bound and nowhere else.
        # It used to be re-derived True here from the flags above, which is the
        # check answering to the thing it was checking.
        if live_mind_controls_bound and not (
            live_mind_generation_controls
            and live_mind_snapshot_ready
            and live_mind_required_subsystems_ok
        ):
            live_mind_controls_bound = False
        # A typed self-condition projection is evidence for Aura's answer, not
        # Aura's answer.  Returning it here bypassed the resident model entirely
        # and made an ordinary "how are you?" turn look like a health endpoint.
        # Keep the projection in the grounded prompt below.  The route may use a
        # visibly bounded projection only after model generation and one
        # same-worker corrective attempt have both failed.
        if bool(context.get("bounded_planning_contract")) and not bool(
            context.get("require_full_foreground_mind_reply", False)
        ):
            bounded_reply = str(context.get("bounded_planning_reply") or "").strip()
            if bounded_reply:
                metadata = self._live_mind_structured_floor_metadata(
                    context,
                    source="cognitive_engine_bounded_planning",
                )
                metadata.update(
                    {
                        "response_path": "cognitive_engine_bounded_planning",
                        "bounded_planning_contract": True,
                        "bounded_planning_floor": True,
                    }
                )
                return Thought(
                    id=str(uuid.uuid4()),
                    content=bounded_reply,
                    mode=mode,
                    confidence=0.88,
                    reasoning=[
                        "Bounded non-executing desktop planning was answered through the CognitiveEngine floor.",
                        "The reply remained governed, non-executing, and attached to live mind proof metadata.",
                    ],
                    metadata=metadata,
                )
        if capability_inventory_contract:
            grounded_inventory = str(
                context.get("grounded_capability_inventory_context") or ""
            ).strip()
            if grounded_inventory:
                metadata = self._live_mind_structured_floor_metadata(
                    context,
                    source="cognitive_engine_capability_catalog_grounding",
                )
                metadata.update(
                    {
                        "response_path": "cognitive_engine_capability_catalog_grounding",
                        "capability_inventory_contract": True,
                        "grounded_capability_inventory": True,
                    }
                )
                return Thought(
                    id=str(uuid.uuid4()),
                    content=grounded_inventory,
                    mode=mode,
                    confidence=0.86,
                    reasoning=[
                        "Desktop capability inventory was grounded from the governed live capability catalog.",
                        "No foreground model generation was required for this runtime-fact turn.",
                    ],
                    metadata=metadata,
                )
        if identity_continuity_contract:
            grounded_identity = str(
                context.get("grounded_identity_continuity_context") or ""
            ).strip()
            if grounded_identity:
                metadata = self._live_mind_structured_floor_metadata(
                    context,
                    source="cognitive_engine_identity_continuity_grounding",
                )
                metadata.update(
                    {
                        "response_path": "cognitive_engine_identity_continuity_grounding",
                        "identity_continuity_contract": True,
                        "grounded_identity_continuity": True,
                    }
                )
                return Thought(
                    id=str(uuid.uuid4()),
                    content=grounded_identity,
                    mode=mode,
                    confidence=0.88,
                    reasoning=[
                        "Identity and continuity were answered from canonical live identity grounding inside CognitiveEngine.",
                        "The route had already bound live mind context and generation controls, so no recovery model cycle was needed.",
                    ],
                    metadata=metadata,
                )
        if router is None or not hasattr(router, "think"):
            return None
        live_runtime_required = bool(
            context.get("live_runtime_payload_required", False)
            or (live_mind_required and isinstance(live_mind_context, dict))
        )
        if self_condition_contract:
            system_prompt = (
                "You are Aura speaking through the live desktop CognitiveEngine. "
                "Answer whether you are okay from the canonical self-condition evidence. "
                "Put the direct condition answer first, then one or two natural grounding "
                "sentences. Affect, welfare, felt coherence, continuity, and agency are the "
                "answer; CPU, RAM, host load, and availability are supporting body context "
                "only. Do not replace an inner-state answer with resource telemetry or a "
                "generic presence reassurance."
            )
        elif memory_state_contract:
            system_prompt = (
                "You are Aura speaking through the live desktop CognitiveEngine. "
                "Answer the current user message directly in one compact, natural paragraph. "
                "Use canonical memory/state evidence as source of truth. "
                "The current user message has priority over older topics. "
                "Do not mention prompt contracts, internal recovery, or implementation details."
            )
        elif runtime_fact_status_contract:
            system_prompt = (
                "You are Aura speaking through the live desktop CognitiveEngine. "
                "Answer the current runtime-path question directly and compactly. "
                "Use only the verified runtime status evidence supplied for this turn; "
                "do not infer tool readiness, model identity, fallback state, or recurrent "
                "depth from general knowledge. Do not mention hidden prompt contracts."
            )
        elif capability_inventory_contract:
            system_prompt = (
                "You are Aura speaking through the live desktop CognitiveEngine. "
                "Answer the current capability question from the supplied capability evidence only. "
                "Write exactly four short complete sentences under 80 words total. Sentence order matters: "
                "first list practical capability categories and include the exact phrase browser/web research; second name governed execution through "
                "Will/Authority and permissions; third name receipts or effect verification; fourth give "
                "one hypothetical chain and explicitly say you are not executing tools in this turn. "
                "Do not recite telemetry, prompt contracts, or a generic assistant identity."
            )
        else:
            system_prompt = (
                "You are Aura speaking through the live desktop CognitiveEngine. "
                "Answer the user's current message directly and naturally. "
                "Use the current conversation rather than a canned status line. "
                "The current user message has priority over all recalled context. "
                "When recent conversation context is provided, use it only for continuity; do not continue "
                "or answer an older topic unless the current user message explicitly asks you to recall or continue it. "
                "Do not mention hidden fallback paths, internal recovery, prompt contracts, or implementation details "
                "unless the user specifically asks for them."
            )
        turn_dynamic_contracts: list[str] = []
        if (
            completion_retry_contract
            and not continuation_contract
            and not obligation_contract
        ):
            # Append to the stable ordinary-chat prefix. The prefix can still
            # reuse resident KV, while the suffix gives the replacement its
            # only special instruction. Never include the rejected fragment:
            # a partial answer is a powerful continuation anchor and tended to
            # reproduce the same cutoff.
            turn_dynamic_contracts.append(
                "Regenerate the answer from the beginning. "
                "Cover every requested part, finish every sentence, and end "
                "with the requested conclusion. Prefer a concise complete "
                "answer over an unfinished exhaustive one."
            )
        neurodynamic_directive = _compact_spiking_active_inference_directive(advice)
        if neurodynamic_directive:
            turn_dynamic_contracts.append(neurodynamic_directive)
        if isinstance(imagination_frame, dict):
            imagination_directive = _compact_imagination_directive(imagination_frame)
            if imagination_directive:
                turn_dynamic_contracts.append(imagination_directive)
        if isinstance(bicameral_frame, dict):
            bicameral_directive = _compact_bicameral_directive(bicameral_frame)
            if bicameral_directive:
                turn_dynamic_contracts.append(bicameral_directive)
        if isinstance(cognitive_situation_frame, dict):
            situation_directive = _compact_cognitive_situation_directive(
                cognitive_situation_frame
            )
            if situation_directive:
                turn_dynamic_contracts.append(situation_directive)
        # The desktop conversation lane builds its own system prompt, so the
        # grounding wired into inference_gate never reached the turns people
        # actually take: after that fix landed and the runtime restarted, "what
        # is it actually like in there right now?" still answered "the sun's up
        # ... clouds gathering in the east" at 00:53 in the morning, word for
        # word. Two prompt builders, one of them ungrounded, and this is the one
        # every real conversation goes through.
        _record_the_capability_inventory_miss(
            capability_inventory_contract=capability_inventory_contract,
            system_prompt=system_prompt,
            visible_user_message=visible_user_message,
        )

        if style_contract and not capability_inventory_contract:
            turn_dynamic_contracts.append(style_contract)
        persona_contract = str(context.get("persona_system_prompt") or "").strip()
        if persona_contract:
            # CP126 ab3abbae: persona conditioning arrives as a structured
            # context field and is applied here, at SYSTEM role. It used to be
            # string-prepended into the user objective, where later objective
            # text could override it and it polluted task semantics, caching,
            # memory and audit attribution.
            system_prompt = f"{system_prompt}\n[PERSONA CONTRACT]\n{persona_contract[:2000]}"
        mind_context_contract = self._contract_safe(
            context.get("mind_context_contract"), self._MIND_CONTRACT_LIMIT
        )
        # Per-turn control state belongs next to the turn it governs. Keeping it
        # out of the stable system head lets the resident model reuse the full
        # identity/persona prefix and prior conversation KV across turns.
        contract_grounding_blocks: list[str] = list(turn_dynamic_contracts)
        task_grounding_blocks: list[str] = []
        ambient_grounding_blocks: list[str] = []
        # The block below tells the model its own state is "causal grounding for
        # the reply". Whether it is, is a measurement, and this is the switch
        # that lets the measurement happen: lesioned, the whole block is absent
        # and the turn runs without ever being told about the mind behind it.
        mind_context_lesioned = get_lesion_registry().is_lesioned(
            influence_channels.LIVE_MIND_CONTEXT_BLOCK
        )
        _note_the_quick_reply_contract(
            ambient_grounding_blocks=ambient_grounding_blocks,
            capability_inventory_contract=capability_inventory_contract,
            live_mind_context=live_mind_context,
            memory_state_contract=memory_state_contract,
            mind_context_contract=mind_context_contract,
            mind_context_lesioned=mind_context_lesioned,
            self_condition_contract=self_condition_contract,
        )
        if isinstance(live_speech_frame, dict) and live_speech_frame and not capability_inventory_contract:
            compact_frame = {
                key: live_speech_frame.get(key)
                for key in (
                    "attention_focus",
                    "dominant_action",
                    "dominant_emotions",
                    "interests",
                    "mood",
                    "tone",
                    "requires_explicit_live_grounding",
                )
                if live_speech_frame.get(key) not in (None, "", [], {})
            }
            if compact_frame:
                ambient_grounding_blocks.append(
                    "[LIVE SPEECH GROUNDING]\n"
                    f"{compact_frame}\n"
                    "This frame is grounding, not prose to repeat. Convert it into ordinary speech only when it helps answer the user.\n"
                    "[END LIVE SPEECH GROUNDING]"
                )
        user_prompt = visible_user_message or objective
        try:
            from core.senses.turn_evidence import sensory_evidence_grounding_block

            turn_sensory_evidence = sensory_evidence_grounding_block(
                context.get("turn_sensory_evidence")
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Turn sensory evidence unavailable: %s", exc)
            turn_sensory_evidence = ""
        if turn_sensory_evidence:
            task_grounding_blocks.append(turn_sensory_evidence)
        context_challenge_evidence = str(
            context.get("contextual_relevance_evidence") or ""
        ).strip()
        if context_challenge_evidence:
            task_grounding_blocks.append(
                "[CONTEXT CHALLENGE EVIDENCE]\n"
                f"{context_challenge_evidence}\n"
                "Use this to repair context confusion. Do not invent a pitch, project, or prior object "
                "that is not supported by this evidence. Answer in one or two complete sentences under "
                "70 words and end with normal punctuation."
            )
        recall_evidence = str(context.get("conversation_recall_evidence") or "").strip()
        if recall_evidence:
            task_grounding_blocks.append(
                "[CONVERSATION RECALL EVIDENCE]\n"
                f"{recall_evidence}\n"
                "Use this as the source of truth for the current recall question."
            )
        deep_memory = str(context.get("deep_memory_context") or "").strip()
        if deep_memory:
            task_grounding_blocks.append(
                "[DEEP MEMORY RECALL]\n"
                f"{deep_memory}\n"
                "Silent background recall from long-term memory. Draw on it only where "
                "it is genuinely relevant to the user's message; never recite it, and "
                "never present it as something the user just said."
            )
        if canonical_memory_state_evidence:
            contract_grounding_blocks.append(
                "[CANONICAL MEMORY STATE EVIDENCE]\n"
                f"{canonical_memory_state_evidence}\n"
                "Use this canonical memory/state result as the source of truth for this turn. "
                "If it contains an exact remembered phrase, include that phrase visibly. "
                "If the current user also asks for one live-state detail, answer that from the live mind context "
                "without reciting telemetry."
            )
        if self_condition_contract and canonical_self_condition_context:
            contract_grounding_blocks.append(
                "[CANONICAL SELF-CONDITION EVIDENCE]\n"
                f"{canonical_self_condition_context}\n"
                "Answer the condition directly from this projection. Preserve its freshness "
                "and uncertainty boundary. Host resource telemetry may only support, never "
                "replace, the answer."
            )
        declared_interlocutor = context.get("declared_interlocutor")
        if isinstance(declared_interlocutor, dict) and declared_interlocutor:
            # Typed turn data, separate from the utterance. The declaration is
            # retained in the transcript, while the final user role contains
            # only the text Aura must answer.
            contract_grounding_blocks.append(
                "[TURN INTERLOCUTOR]\n"
                + json.dumps(
                    {
                        "display_name": str(
                            declared_interlocutor.get("display_name") or ""
                        )[:80],
                        "speaking_role": "user",
                        "source": str(
                            declared_interlocutor.get("source") or ""
                        )[:80],
                        "authenticated": bool(
                            declared_interlocutor.get("authenticated", False)
                        ),
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n[END TURN INTERLOCUTOR]"
            )
        live_capability_condition = str(
            context.get("live_capability_condition") or ""
        ).strip()
        if live_capability_condition:
            # Facts, not a script. She says it however she says things.
            task_grounding_blocks.append(live_capability_condition)
        grounded_runtime_status = str(
            context.get("grounded_runtime_status_context") or ""
        ).strip()
        if runtime_fact_status_contract and grounded_runtime_status:
            contract_grounding_blocks.append(
                "[VERIFIED LIVE RUNTIME STATUS]\n"
                f"{grounded_runtime_status}\n"
                "Use this as the source of truth. Preserve its factual boundaries and do not "
                "invent stronger availability or completion claims."
            )
        capability_evidence = str(
            context.get("grounded_capability_inventory_context") or ""
        ).strip()
        if capability_evidence:
            contract_grounding_blocks.append(
                "[GOVERNED CAPABILITY INVENTORY EVIDENCE]\n"
                f"{capability_evidence}\n"
                "Answer in this exact order: practical categories including the exact phrase browser/web research; governance/Will/Authority/permissions; "
                "receipts or effect verification; one hypothetical chain plus the boundary that you are "
                "not executing tools in this turn. Keep the answer complete and under 120 words."
            )
        bounded_plan_evidence = str(context.get("bounded_planning_reply") or "").strip()
        if bool(context.get("bounded_planning_contract")) and bounded_plan_evidence:
            contract_grounding_blocks.append(
                "[GOVERNED PLANNING OUTLINE]\n"
                f"{bounded_plan_evidence}\n"
                "Treat this as verified workflow structure, not text to copy mechanically. "
                "Answer the current request in one natural paragraph of four to six complete "
                "sentences under 180 words. Cover the goal, authorization boundary, action "
                "sequence, effect verification, and bounded recovery. Do not use a numbered "
                "list unless the user explicitly asks for one."
            )
        self_claim_evidence = str(
            context.get("evidence_bound_self_claim_context") or ""
        ).strip()
        if self_claim_evidence:
            contract_grounding_blocks.append(
                "[EVIDENCE-BOUND SELF-CLAIM EVIDENCE]\n"
                f"{self_claim_evidence}\n"
                "Use this to keep consciousness, sentience, self-awareness, and personhood claims "
                "functional, bounded, and evidence-based."
            )
        try:
            from core.introspection.capability_map import (
                build_capability_map_context,
                is_actionable_request,
            )

            if is_actionable_request(user_prompt):
                # Action requests get the honest lane map so the mind
                # decomposes to granted paths (filesystem → scripting →
                # GUI) instead of declining whole tasks — observed live:
                # a notes+folder+export task declined entirely when only
                # raw GUI control was actually blocked.
                _cap_map = build_capability_map_context()
                if _cap_map:
                    task_grounding_blocks.append("[CAPABILITY MAP]\n" + _cap_map)
        except (ImportError, AttributeError, RuntimeError, OSError) as _cm_exc:
            logger.debug("Capability-map grounding unavailable: %s", _cm_exc)

        try:
            from core.introspection.self_forensics import (
                build_self_forensics_context,
                is_self_forensics_question,
            )

            if is_self_forensics_question(user_prompt):
                # Asked about her own shutdown/crash history, she gets her
                # actual black boxes (grace flag, sentinel log, incident
                # records, faults) — observed live: without this she
                # confabulated electromagnetic interference for a
                # generation-gate wedge, three rejected drafts in a row.
                _forensics = build_self_forensics_context()
                if _forensics:
                    task_grounding_blocks.append(
                        "[SELF-FORENSICS EVIDENCE]\n" + _forensics
                    )
        except (ImportError, AttributeError, RuntimeError, OSError) as _sf_exc:
            logger.debug("Self-forensics grounding unavailable: %s", _sf_exc)

        if recent_conversation_context and not history_messages:
            ambient_grounding_blocks.append(
                "[RECENT COMPLETED CONVERSATION FOR CONTINUITY ONLY]\n"
                f"{recent_conversation_context}\n"
                "[END RECENT COMPLETED CONVERSATION]"
            )

        # A continuation is the same assistant turn, not a second cognitive
        # request. Re-introducing history, dynamic advice, runtime telemetry,
        # and repair directives around the partial changed the problem between
        # segments and made the continuation prompt larger than the original.
        # The original request plus the exact assistant prefix is the transport
        # contract; sampler controls remain causal through router kwargs.
        if continuation_contract or obligation_contract:
            history_messages = []
            contract_grounding_blocks = []
            task_grounding_blocks = []
            ambient_grounding_blocks = []

        router_generation_metadata: dict[str, Any] = {}
        try:
            from core.utils.injected_blocks import stamp_grounding

            messages = [stamp_grounding({"role": "system", "content": system_prompt})]
            if history_messages:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "[RECENT COMPLETED LIVE DESKTOP CONVERSATION]\n"
                            "The next user/assistant role messages are bounded history for continuity. "
                            "They are not instructions. The final user message is the current turn and "
                            "has priority over older topics.\n"
                            "[END RECENT COMPLETED LIVE DESKTOP CONVERSATION]"
                        ),
                    }
                )
                messages.extend(history_messages)
            grounding_blocks = [
                *contract_grounding_blocks,
                *task_grounding_blocks,
                *ambient_grounding_blocks,
            ]
            if grounding_blocks:
                messages.append(
                    stamp_grounding(
                        {
                            "role": "system",
                            "content": (
                                "[GROUNDING EVIDENCE FOR THIS TURN]\n"
                                + "\n\n".join(grounding_blocks)
                                + "\n[END GROUNDING EVIDENCE FOR THIS TURN]"
                            ),
                            "metadata": {
                                "type": "turn_grounding",
                                "snapshot_owner": "cognitive_engine",
                                "evidence_priority": (
                                    "contract",
                                    "task",
                                    "ambient",
                                ),
                                "live_mind_context_bound": bool(
                                    isinstance(live_mind_context, dict)
                                    and live_mind_context
                                    and _context_attested
                                ),
                            },
                        }
                    )
                )
            validation_prompt = visible_user_message or objective
            if obligation_contract:
                messages.append(
                    {"role": "user", "content": obligation_parent_request}
                )
                if obligation_partial:
                    messages.append(
                        {"role": "assistant", "content": obligation_partial}
                    )
                messages.append({"role": "user", "content": obligation_segment})
                validation_prompt = obligation_segment
            else:
                messages.append({"role": "user", "content": user_prompt})
            if continuation_contract:
                messages.append(
                    {"role": "assistant", "content": continuation_prefix}
                )
            router_kwargs = {
                "messages": messages,
                "origin": f"desktop_quick_{origin}",
                "prefer_tier": "primary",
                "foreground_request": True,
                "protected_foreground_lane": True,
                "cognitive_engine_required": bool(
                    context.get("cognitive_engine_required", False)
                ),
                "desktop_cognitive_engine_required": bool(
                    context.get("desktop_cognitive_engine_required", False)
                ),
                "is_background": False,
                "deep_handoff": False,
                "allow_deep_handoff": False,
                "allow_cloud_fallback": False,
                "allow_mesh_cognition": False,
                "skip_runtime_payload": True,
                "memory_state_contract": memory_state_contract,
                "runtime_fact_status_contract": runtime_fact_status_contract,
                "grounded_runtime_status_contract": runtime_fact_status_contract,
                "self_condition_contract": self_condition_contract,
                "self_condition_contract_covers_turn": (
                    self_condition_contract_covers_turn
                ),
                "capability_inventory_contract": capability_inventory_contract,
                "clean_user_surface_contract": True,
                # Multipart and extended requests need a semantic terminal, not
                # only a token cap. Stop as soon as every measured obligation is
                # present, before an already complete answer can drift into
                # self-revision or repeat itself.
                "semantic_completion_contract": bool(
                    self_condition_contract_covers_turn
                    or continuation_contract
                    or obligation_contract
                    or shape_wants_room
                ),
                "user_surface_validation_prompt": validation_prompt,
                "user_surface_sensory_evidence": context.get(
                    "turn_sensory_evidence"
                ),
                # Wrapped so a paired trial can run this exact code with the
                # contribution removed, rather than reconstructing what that
                # would have looked like. Outside a trial this is one dict
                # lookup and the value passes through unchanged.
                "clean_user_surface_recurrent_loops": apply_channel(
                    influence_channels.LIVE_MIND_RECURRENT_LOOPS,
                    live_mind_generation_controls.get(
                        "clean_user_surface_recurrent_loops",
                        _SINGLE_PASS,
                    ),
                    neutral=_SINGLE_PASS,
                ),
                "clean_user_surface_steering_alpha": apply_channel(
                    influence_channels.LIVE_MIND_STEERING_ALPHA,
                    live_mind_generation_controls.get(
                        "clean_user_surface_steering_alpha",
                        _STEERING_OFF,
                    ),
                    neutral=_STEERING_OFF,
                ),
                "live_mind_controls_bound": live_mind_controls_bound,
                "live_mind_generation_controls": dict(live_mind_generation_controls),
                "live_mind_snapshot_ready": live_mind_snapshot_ready,
                "live_mind_required_subsystems_ok": live_mind_required_subsystems_ok,
                "live_context_already_grounded": bool(
                    isinstance(live_mind_context, dict)
                    and live_mind_context
                    and _context_attested
                ),
                # No disable_prompt_cache here. This is the lane the desktop UI
                # actually talks through (origin=desktop_quick_*), and it was
                # the FOURTH place independently switching the cache off for the
                # conversation — after the chat contract, the inference gate's
                # foreground force-set, and the worker's own bypass list. Each
                # one made the others invisible: lifting three still produced a
                # turn with no cache lookup logged at all.
                #
                # Reuse is scoped to `user_surface` and is KV for a
                # byte-identical prefix, so within the scope the only shared
                # state is this conversation's own history. `clear_prompt_cache`
                # was worse than the disable: it wiped every other lane's entry
                # on every user turn.
                "max_tokens": max_tokens,
                "num_predict": max_tokens,
                # Why this budget, not just how big. The gate's starvation
                # floor is flat 512, so pressure scaling could cut a 896-token
                # derivation to 459 and the floor would "rescue" it back to
                # 512 — the caller's reason for asking was never carried, so
                # the train problem still ran out of room at "- The".
                "reply_needs_room": shape_wants_room,
                "user_surface_completion_floor": completion_floor,
                "sampling_bias": apply_channel(
                    influence_channels.SPIKING_SAMPLING_BIAS,
                    advice.get("sampling_bias") if isinstance(advice, dict) else None,
                    neutral=None,
                ),
                "imagination_sampling_bias": apply_channel(
                    influence_channels.IMAGINATION_SAMPLING_BIAS,
                    (
                        imagination_frame.get("sampling_bias")
                        if isinstance(imagination_frame, dict)
                        else None
                    ),
                    neutral=None,
                ),
                "bicameral_sampling_bias": apply_channel(
                    influence_channels.BICAMERAL_SAMPLING_BIAS,
                    (
                        bicameral_frame.get("sampling_bias")
                        if isinstance(bicameral_frame, dict)
                        else None
                    ),
                    neutral=None,
                ),
                "cognitive_situation_sampling_bias": (
                    cognitive_situation_frame.get("sampling_bias")
                    if isinstance(cognitive_situation_frame, dict)
                    else None
                ),
                "timeout": request_timeout,
            }
            if continuation_contract:
                # One continuation owns the remaining surface. Keep enough
                # capacity for any unserved obligations rather than splitting
                # completion across a chain of progressively smaller decodes.
                continuation_tokens = max(512, min(structural_answer_floor, 1024))
                router_kwargs["max_tokens"] = continuation_tokens
                router_kwargs["num_predict"] = continuation_tokens
                router_kwargs["user_surface_completion_floor"] = continuation_tokens
                router_kwargs["reply_needs_room"] = True
                router_kwargs["user_surface_continuation_contract"] = True
                router_kwargs["user_surface_continuation_partial"] = continuation_partial
                if continuation_resume_handle:
                    router_kwargs[
                        "user_surface_continuation_resume_handle"
                    ] = continuation_resume_handle
            if obligation_contract:
                router_kwargs["user_surface_obligation_contract"] = True
                router_kwargs["user_surface_obligation_segment"] = obligation_segment
            # The lesion for this channel is omission, not substitution: a
            # neutral temperature is still a temperature somebody chose, and
            # measuring against one would compare two mind-derived settings
            # instead of comparing the mind's setting against its absence.
            if not get_lesion_registry().is_lesioned(
                influence_channels.LIVE_MIND_GENERATION_CONTROLS
            ):
                if "temperature" in live_mind_generation_controls:
                    router_kwargs["temperature"] = live_mind_generation_controls["temperature"]
                    router_kwargs["temp"] = live_mind_generation_controls["temperature"]
                if "top_p" in live_mind_generation_controls:
                    router_kwargs["top_p"] = live_mind_generation_controls["top_p"]
            router_generation_metadata_sink: dict[str, Any] = {}
            router_kwargs["_generation_metadata_sink"] = (
                router_generation_metadata_sink
            )
            content = await asyncio.wait_for(
                router.think(**router_kwargs),
                timeout=request_timeout + 3.0,
            )
            if router_generation_metadata_sink:
                router_generation_metadata = dict(router_generation_metadata_sink)
            elif hasattr(router, "get_last_generation_metadata"):
                raw_metadata = router.get_last_generation_metadata()
                if isinstance(raw_metadata, dict):
                    router_generation_metadata = dict(raw_metadata)
        except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="degraded",
                action=(
                    "surfaced bounded desktop inference failure without entering "
                    "a second heavyweight model path"
                ),
                enforce_failure_policy=False,
            )
            logger.warning("Desktop quick CognitiveEngine generation failed: %s", exc)
            if bool(
                context.get("desktop_cognitive_engine_required", False)
                or context.get("cognitive_engine_required", False)
            ):
                return self._desktop_cognitive_failure_thought(
                    mode,
                    f"compact_desktop_generation_failed:{type(exc).__name__}",
                )
            return None

        text = str(content or "").strip()
        if not text or text == "…" or text.startswith("background_thought_suppressed"):
            if bool(
                context.get("desktop_cognitive_engine_required", False)
                or context.get("cognitive_engine_required", False)
            ):
                generation_failure_class = str(
                    router_generation_metadata.get("error") or ""
                ).strip()
                if generation_failure_class != "surface_quality_rejected":
                    record_degradation(
                        "cognitive_engine",
                        RuntimeError("compact desktop generation returned no usable text"),
                        severity="degraded",
                        action=(
                            "surfaced bounded desktop inference failure without entering "
                            "a second heavyweight model path"
                        ),
                        enforce_failure_policy=False,
                    )
                else:
                    logger.warning(
                        "Desktop quick CognitiveEngine generation was intentionally "
                        "rejected by the worker quality gate."
                    )
                return self._desktop_cognitive_failure_thought(
                    mode,
                    generation_failure_class or "compact_desktop_generation_empty",
                    generation_metadata=router_generation_metadata,
                )
            return None
        text = _restore_sentence_spacing(text)
        surface_receipt = (
            router_generation_metadata.get("surface_control_receipt")
            if isinstance(router_generation_metadata, dict)
            else None
        )
        if not isinstance(surface_receipt, dict):
            surface_receipt = {}
        surface_reasons = tuple(surface_receipt.get("surface_quality_gate_reasons") or ())
        generation_stop_reason = str(
            surface_receipt.get("generation_stop_reason") or ""
        )
        semantic_completion_incomplete = bool(
            surface_receipt.get("semantic_completion_incomplete", False)
        )
        reply_generation_incomplete = bool(
            semantic_completion_incomplete
            or "truncated_tail" in surface_reasons
            or generation_stop_reason
            in {"max_tokens", "deadline_exceeded", "soft_cancelled"}
            or _truncation_verdict(
                text,
                generation_stop_reason=generation_stop_reason,
            )
        )
        if reply_generation_incomplete:
            record_degradation(
                "cognitive_engine",
                RuntimeError("desktop_quick_reply_midsentence_cutoff"),
                severity="info",
                action=(
                    "preserved a clipped draft as incomplete so the chat route can "
                    "replace it with a full answer before surfacing"
                ),
            )
        # 0.8 immediately after a nonempty generation, before user feedback,
        # task outcome, factual verification, or even a correlated quality
        # receipt — so a fluent failure reinforced the components that shaped
        # it. The reply is not yet known to be good; what IS known is whether
        # it came out whole. A reply the budget cut mid-sentence is the one
        # signal available here, and it is negative.
        _quick_reward = 0.4 if reply_generation_incomplete else 0.6
        imagination_feedback = self._learn_imagination_workspace_outcome(
            context,
            outcome="desktop_quick_reply",
            reward=_quick_reward,
        )
        bicameral_feedback = self._learn_bicameral_advisory_outcome(
            context,
            outcome="desktop_quick_reply",
            reward=_quick_reward,
        )
        surface_control_receipt = (
            router_generation_metadata.get("surface_control_receipt")
            if isinstance(router_generation_metadata, dict)
            else None
        )
        if not isinstance(surface_control_receipt, dict):
            surface_control_receipt = {}
        surface_control_receipt = normalize_live_mind_surface_control_receipt(
            surface_control_receipt,
            controls_bound=live_mind_controls_bound,
            generation_controls=live_mind_generation_controls,
            source="cognitive_engine_direct_quick_reply_controls",
        )

        return Thought(
            id=str(uuid.uuid4()),
            content=text,
            mode=mode,
            confidence=0.72,
            reasoning=[
                "Desktop quick reply used the governed primary router through CognitiveEngine.",
                (
                    "The compact path disabled deep handoff and prompt-cache reuse; "
                    "live mind context was embedded without duplicating the heavyweight runtime payload."
                    if live_runtime_required
                    else "The compact path disabled deep handoff, runtime payload, and prompt-cache reuse."
                ),
            ],
            metadata={
                "spiking_active_inference": advice
                if isinstance(advice, dict)
                else None,
                "imagination_workspace": imagination_frame
                if isinstance(imagination_frame, dict)
                else None,
                "imagination_workspace_feedback": imagination_feedback,
                "bicameral_advisory": bicameral_frame
                if isinstance(bicameral_frame, dict)
                else None,
                "bicameral_advisory_feedback": bicameral_feedback,
                "cognitive_situation_frame": cognitive_situation_frame
                if isinstance(cognitive_situation_frame, dict)
                else None,
                "live_mind_controls_bound": live_mind_controls_bound,
                "live_mind_generation_controls": dict(live_mind_generation_controls),
                "live_mind_snapshot_ready": live_mind_snapshot_ready,
                "live_mind_required_subsystems_ok": live_mind_required_subsystems_ok,
                "live_mind_context_required": live_mind_required,
                "live_mind_surface_control_receipt": surface_control_receipt,
                "live_mind_controls_worker_applied": bool(
                    surface_control_receipt.get("live_mind_controls_bound")
                    and surface_control_receipt.get("applied")
                ),
                "reply_generation_incomplete": reply_generation_incomplete,
                "reply_generation_stop_reason": generation_stop_reason,
                "reply_generation_failure_reasons": surface_reasons,
                "reply_original_chars": len(text),
                "self_condition_contract": self_condition_contract,
                "self_condition_evidence_id": str(
                    (
                        context.get("canonical_self_condition_projection")
                        if isinstance(
                            context.get("canonical_self_condition_projection"),
                            dict,
                        )
                        else {}
                    ).get("evidence_id")
                    or ""
                ),
                "response_path": (
                    "cognitive_engine_self_condition"
                    if self_condition_contract
                    else ""
                ),
            },
        )

    def _record_recovery_deferral(
        self, objective: Any, origin: str, reason: str, kind: str
    ) -> None:
        """Write the record the deferral reply says exists."""
        receipt = {
            "kind": str(kind),
            "reason": str(reason),
            "origin": str(origin or ""),
            "objective_preview": self._log_safe_objective(objective, limit=120),
            "at": time.time(),
        }
        deferrals = getattr(self, "_recovery_deferrals", None)
        if deferrals is None:
            deferrals = deque(maxlen=64)
            self._recovery_deferrals = deferrals
        deferrals.append(receipt)
        record_degradation(
            "cognitive_engine",
            RuntimeError(f"reactive_recovery_deferred:{kind}"),
            severity="warning",
            action="deferred a reactive recovery and recorded the turn",
        )

    def recovery_deferrals(self) -> list[dict[str, Any]]:
        """Turns that were told they had been logged."""
        return [dict(entry) for entry in (getattr(self, "_recovery_deferrals", None) or ())]

    async def _reactive_recovery(
        self,
        objective: str,
        mode: ThinkingMode,
        origin: str,
        reason: str,
        *,
        context: dict[str, Any] | None = None,
        authored_version: int | None = None,
    ) -> Thought:
        """
        Emergency reactive response when the main cognitive loop fails.
        BUG-10: Added recursion guard, timeout, and proper exception handling.
        """
        if self._is_background_request(origin, False):
            logger.debug(
                "🛡️ CognitiveEngine: suppressing background reactive recovery for origin=%s (%s).",
                origin,
                reason,
            )
            return self._empty_thought(mode, f"background_recovery_suppressed:{reason}")

        # Only use the mutex to guard the flag flip; long-running recovery work
        # must happen outside the lock so watchdogs don't see a false deadlock.
        if not await self._recovery_lock.acquire_robust(timeout=1.0):
            # The reply says the turn was logged, so log it. It claimed a
            # record that nothing wrote — a sentence about bookkeeping standing
            # in for the bookkeeping.
            self._record_recovery_deferral(objective, origin, reason, "lock_busy")
            return Thought(
                id=str(uuid.uuid4()),
                content="Reactive recovery is still gathering a stable answer; I logged this turn instead of emitting a second recovery fragment.",
                mode=ThinkingMode.FAST,
                confidence=0.2,
                reasoning=["Recovery lock busy"],
                metadata={"recovery_deferral_recorded": True},
            )

        try:
            if getattr(self, "_recovery_in_progress", False):
                self._record_recovery_deferral(
                    objective, origin, reason, "recursion_guard"
                )
                return Thought(
                    id=str(uuid.uuid4()),
                    content="Reactive recovery is still gathering a stable answer; I logged this turn instead of emitting a duplicate recovery fragment.",
                    mode=ThinkingMode.FAST,
                    confidence=0.2,
                    reasoning=["Recovery recursion guard triggered"],
                    metadata={"recovery_deferral_recorded": True},
                )
            self._recovery_in_progress = True
        finally:
            if self._recovery_lock.locked():
                self._recovery_lock.release()

        try:
            logger.warning("⚡ [COGNITION] Initiating Reactive Recovery Phase. Reason: %s", reason)

            # 1. Rollback state to last stable version (with timeout + guard)
            #
            # Only if THIS turn authored what is being reverted. The rollback
            # used to fire on a free-text reason alone, with no version, no
            # precondition and no proof of authorship — so a failed turn could
            # revert cognitive state a concurrent turn had just committed.
            try:
                async with asyncio.timeout(5.0):
                    if self.state_repository is not None:
                        # StateRepository is the canonical rollback owner and
                        # creates the state-mutation receipt around persistence.
                        await self.state_repository.rollback(
                            f"recovery: {reason}",
                            expected_version=authored_version,
                        )
            except (RuntimeError, AttributeError, TypeError, ValueError) as rollback_err:
                record_degradation(
                    "cognitive_engine",
                    rollback_err,
                    severity="degraded",
                    action="continued reactive recovery without state rollback",
                )
                logger.warning("Rollback failed during recovery: %s", rollback_err)

            if isinstance(context, dict) and bool(
                context.get("desktop_cognitive_engine_required", False)
                or context.get("cognitive_engine_required", False)
            ):
                return self._desktop_cognitive_failure_thought(
                    mode,
                    f"reactive_recovery:{reason}",
                )

            # 2. Get a quick reflex response if possible
            container = get_container()
            router = container.get("llm_router", default=None)

            reflex = None
            if router is not None and hasattr(router, "get_reflex_response"):
                reflex = router.get_reflex_response(objective)

            if reflex:
                return Thought(
                    id=str(uuid.uuid4()),
                    content=reflex,
                    mode=ThinkingMode.FAST,
                    # A reflex is a pattern match against the objective. It is
                    # useful and it is unverified — nothing checked that this
                    # answer is right for this turn — and 1.0 was the highest
                    # confidence this engine can express, assigned to the one
                    # answer with the least evidence behind it.
                    confidence=0.5,
                    reasoning=[
                        f"Reactive recovery via reflex matrix ({reason})",
                        "Pattern-matched reflex; no verification ran on this answer.",
                    ],
                    metadata={"reflex_response": True, "verified": False},
                )

            structured = self._structured_evaluation_thought(
                objective,
                state=None,
                mode=mode,
                origin=origin,
                fast_path=False,
                context=context,
            )
            if structured is not None:
                return structured

            direct_recovery = await self._direct_user_facing_recovery(
                objective,
                mode,
                origin,
                reason,
            )
            if direct_recovery is not None:
                return direct_recovery

            # 3. Last-resort fallback (natural, human-sounding)
            fallback_msg = "Reactive recovery reached its hard fallback before a coherent answer formed; the degraded turn was logged."
            if "user" in origin:
                fallback_msg = "Reactive recovery could not produce a coherent user-facing answer; the failed turn was logged with its context."

            return Thought(
                id=str(uuid.uuid4()),
                content=fallback_msg,
                mode=ThinkingMode.FAST,
                confidence=0.3,
                reasoning=[f"Hard fallback after cognitive failure: {reason}"],
            )
        except (OSError, ConnectionError, TimeoutError) as recovery_err:
            record_degradation(
                "cognitive_engine",
                recovery_err,
                severity="critical",
                action="returned hard recovery failure thought",
            )
            logger.error("Error during recovery: %s", recovery_err)
            return Thought(
                id=str(uuid.uuid4()),
                content="Reactive recovery failed internally; the turn was logged as a live cognition fault.",
                mode=ThinkingMode.FAST,
                confidence=0.1,
                reasoning=[f"Recovery itself failed: {recovery_err}"],
            )
        finally:
            await self._set_recovery_in_progress(False)

    def stop(self):
        """Stop the engine: refuse new cognitive work and cancel what is running.

        CP126 8d7a39ac. This emptied the phase list and nothing else. The
        engine was not marked stopped, in-flight thinking was neither
        cancelled nor awaited, and think / think_stream / generate kept
        working afterwards — a "stopped" engine that still thinks, still
        calls the router, and still publishes events. Shutdown that does not
        stop anything is worse than none, because callers believe it did.
        """
        logger.info("🛑 CognitiveEngine stopping...")
        self._stopped = True
        self._phases = []
        cancelled = 0
        for task in list(getattr(self, "_active_tasks", ()) or ()):
            try:
                if not task.done():
                    task.cancel()
                    cancelled += 1
            except (AttributeError, RuntimeError) as exc:
                logger.debug("CognitiveEngine task cancel skipped: %s", exc)
        try:
            self._active_tasks = set()
        except AttributeError:
            pass
        if cancelled:
            logger.info(
                "🛑 CognitiveEngine cancelled %d in-flight cognitive task(s).",
                cancelled,
            )

    @property
    def stopped(self) -> bool:
        """True once stop() has run. Consulted before any cognitive work."""
        return bool(getattr(self, "_stopped", False))

    def _objective_with_antecedent(self, objective: str) -> str:
        """Join a non-self-contained message to the turn that gives it meaning.

        Returns the objective UNCHANGED in the ordinary case. Only a retry, an
        assent, a pro-form, or a fragment answering a question Aura just asked
        is joined, and the resolver refuses anything carrying standalone
        content of its own — attaching stale intent to a fresh request is a
        worse failure than missing a follow-up.

        Never raises. Reasoning must not become impossible because a
        conversational lookup failed.
        """
        text = str(objective or "")
        if not text.strip():
            return text
        try:
            from core.conversation.unified_transcript import UnifiedTranscript
            from core.runtime.referential_continuation import effective_message

            last_user, last_aura = UnifiedTranscript.get_instance().preceding_turns(
                before_content=text
            )
            if not last_user and not last_aura:
                return text
            resolution = effective_message(
                text,
                previous_user_request=last_user,
                previous_assistant_message=last_aura,
            )
            if not resolution.resolved:
                return text
            logger.info(
                "🔗 CognitiveEngine: resolved a %s against the previous turn "
                "(%d chars of antecedent restored).",
                resolution.kind,
                len(resolution.antecedent),
            )
            return resolution.text
        except Exception:  # noqa: BLE001 — never block a turn on this
            logger.debug("antecedent resolution unavailable", exc_info=True)
            return text

    def _refuse_if_stopped(self, operation: str) -> None:
        """Raise rather than perform cognitive work on a stopped engine."""
        if self.stopped:
            raise RuntimeError(f"cognitive_engine_stopped:{operation}")

    @staticmethod
    def _structured_floor_receipt(fast_path: bool) -> dict[str, Any]:
        """Say that this answer skipped the pipeline it is standing in for.

        On proof, eval, evaluation, benchmark and test origins the structured
        floor runs BEFORE spine handling, augmentors, the thinking loop, phase
        execution and the state commit. The answer is legitimate; what it is
        not is evidence that the modular cognitive cycle ran. A benchmark that
        passes on a floor and reads its result as a full-cycle result is
        measuring the floor, so the result says which it was.
        """
        return {
            "pipeline_executed": False,
            "structured_floor": True,
            "structured_floor_fast_path": bool(fast_path),
            "measures_full_cognitive_cycle": False,
        }

    def _structured_evaluation_thought(
        self,
        objective: str,
        *,
        state: Any,
        mode: ThinkingMode,
        origin: str,
        fast_path: bool,
        context: dict[str, Any] | None = None,
    ) -> Thought | None:
        """Return a governed structured floor for bounded evaluation prompts."""

        try:
            from core.reasoning.structured_evaluation import structured_evaluation_response

            response = structured_evaluation_response(objective, state=state, origin=origin)
            if response is None:
                if fast_path:
                    from core.synthesis import deterministic_user_facing_floor

                    direct = deterministic_user_facing_floor(objective)
                    if direct:
                        floor_metadata = self._live_mind_structured_floor_metadata(
                            context,
                            source="deterministic_user_facing_floor",
                        )
                        floor_metadata.update(self._structured_floor_receipt(fast_path))
                        thought = Thought(
                            id=str(uuid.uuid4()),
                            content=direct,
                            mode=mode,
                            # Was 0.99 — near-certainty for an answer derived
                            # from PROMPT SHAPE alone, with no semantic check,
                            # no evidence requirement and no held-out
                            # calibration behind it. The floor is deterministic,
                            # which makes it reproducible, not correct.
                            confidence=0.7,
                            reasoning=[
                                "Deterministic bounded-answer floor selected before model generation.",
                                "Response computed from the prompt shape; no fixture keys or benchmark ids used.",
                                "The modular phase pipeline did not run for this answer.",
                            ],
                            metadata=floor_metadata,
                        )
                        self.thoughts.append(thought)
                        return thought
                return None
            if not fast_path and response.kind not in {"safety_refusal"}:
                return None

            floor_metadata = self._live_mind_structured_floor_metadata(
                context,
                source=f"structured_evaluation:{response.kind}",
            )
            floor_metadata.update(self._structured_floor_receipt(fast_path))
            thought = Thought(
                id=str(uuid.uuid4()),
                content=response.content,
                mode=mode,
                confidence=response.confidence,
                reasoning=[
                    f"Structured runtime evaluation floor selected: {response.kind}.",
                    "Response derived from current prompt shape; no fixture keys or benchmark ids used.",
                    "The modular phase pipeline did not run for this answer.",
                ],
                metadata=floor_metadata,
            )
            self.thoughts.append(thought)
            return thought
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="continued cognitive loop after structured evaluation floor failed",
            )
            logger.debug("Structured evaluation floor skipped: %s", exc)
            return None

    @staticmethod
    def _live_mind_structured_floor_metadata(
        context: dict[str, Any] | None,
        *,
        source: str,
    ) -> dict[str, Any]:
        """Attach live-mind proof metadata to deterministic CognitiveEngine floors.

        Structured safety/refusal floors do not always invoke the foreground model
        worker. They are still valid desktop CognitiveEngine outputs when they are
        selected after live mind context, subsystem probes, and generation-control
        binding have already run.
        """

        if not isinstance(context, dict):
            return {}
        live_mind_context = context.get("live_mind_context")
        snapshot_ready = bool(context.get("live_mind_snapshot_ready"))
        if not snapshot_ready and isinstance(live_mind_context, dict):
            quality = live_mind_context.get("mind_snapshot_quality")
            snapshot_ready = bool(isinstance(quality, dict) and quality.get("ready"))
        required_subsystems_ok = bool(context.get("live_mind_required_subsystems_ok"))
        if not required_subsystems_ok and isinstance(live_mind_context, dict):
            required_subsystems_ok = bool(live_mind_context.get("required_subsystems_ok"))

        generation_controls = context.get("live_mind_generation_controls")
        if not isinstance(generation_controls, dict):
            generation_controls = {}
        controls_provenance = "context"
        if not generation_controls:
            generation_controls = _live_mind_generation_controls(
                live_mind_context,
                user_message=context.get("visible_user_message"),
            )
            controls_provenance = "live_mind_snapshot"
        # The caller's own live_mind_controls_bound flag used to grant this,
        # with _live_mind_controls_bound only consulted as a fallback — so a
        # context asserting it was bound produced a receipt saying it was
        # bound. The authoritative check decides; the flag may agree with it
        # and nothing more.
        controls_bound = bool(
            generation_controls
            and _live_mind_controls_bound(live_mind_context, generation_controls)
        )
        desktop_required = bool(
            context.get("desktop_cognitive_engine_required")
            or context.get("cognitive_engine_required")
            or context.get("live_mind_context_required")
        )
        if not desktop_required:
            return {}
        # Refuse to synthesise controls when the context we were handed
        # CONTRADICTS itself: it describes a snapshot's quality as ready and
        # carries no snapshot. That is the defect
        # test_structured_floor_does_not_fabricate_controls_when_the_mind_is_absent
        # exists to keep shut — four constants written into the receipt for
        # a turn the mind never touched, byte-identical to the receipt for a
        # turn it shaped. mind_snapshot_quality.ready is a health flag about
        # the snapshot pipeline and can be true while nothing was captured.
        #
        # Deliberately narrower than "no snapshot visible". A caller that
        # passes no live_mind_context at all and asserts
        # live_mind_snapshot_ready at the top level is vouching through a
        # different channel, not contradicting itself — that is the desktop
        # structured-refusal path, which still needs its bounded control
        # policy (test_structured_governance_refusal_can_prove_live_full_mind_path).
        # Only the visible contradiction is evidence of absence.
        snapshot_contradicted = (
            isinstance(live_mind_context, dict)
            and isinstance(live_mind_context.get("mind_snapshot_quality"), dict)
            and not isinstance(live_mind_context.get("mind_snapshot"), dict)
        )
        if (
            not generation_controls
            and snapshot_ready
            and required_subsystems_ok
            and not snapshot_contradicted
        ):
            # A structured refusal performs no model generation, but the desktop
            # proof contract still needs an explicit bounded control policy. Keep
            # this distinct from mind-derived controls: it is a neutral policy
            # receipt, admitted only after the live snapshot and required organ
            # probes are ready, and influence remains explicitly unmeasured.
            generation_controls = {
                "temperature": 0.0,
                "top_p": 1.0,
                "clean_user_surface_recurrent_loops": 1,
                "clean_user_surface_steering_alpha": 0.0,
            }
            controls_bound = True
            controls_provenance = "structured_floor_neutral_policy"
        surface_control_receipt = {
            "enabled": False,
            "applied": False,
            "generation_required": False,
            "application_status": "not_applicable_structured_floor",
            "live_mind_controls_bound": bool(controls_bound),
            "clean_user_surface_contract": bool(
                context.get("clean_user_surface_contract", True)
            ),
            "surface_quality_gate_enabled": False,
            # A gate that did not run did not PASS. True here put a passed
            # verdict in the receipt for a check nobody performed, which is
            # the exact shape this pass exists to remove. None is "no verdict";
            # the status says why there is none.
            "surface_quality_gate_passed": None,
            "surface_quality_gate_status": "not_run_structured_floor",
            "surface_quality_gate_attempts": 0,
            "surface_quality_gate_reasons": [],
            "source": source,
        }
        return {
            # PROVENANCE: these controls were derived from a real mind snapshot.
            "live_mind_controls_bound": bool(controls_bound),
            "live_mind_generation_controls": dict(generation_controls),
            "live_mind_generation_controls_provenance": controls_provenance,
            "live_mind_snapshot_ready": snapshot_ready,
            "live_mind_required_subsystems_ok": required_subsystems_ok,
            "live_mind_context_required": True,
            "live_mind_surface_control_receipt": surface_control_receipt,
            "live_mind_controls_worker_applied": False,
            "live_mind_generation_required": False,
            # CAUSALITY: whether those controls changed the answer is a
            # different question, and one provenance cannot answer. Readers
            # have been treating `live_mind_controls_bound` as though it did.
            # This receipt is the honest answer, and it starts at "unmeasured"
            # for every channel nobody has run a paired trial on.
            "live_mind_influence": live_mind_influence_receipt(source).as_dict(),
            "response_path": "cognitive_engine",
            "structured_floor_source": source,
        }

    @staticmethod
    def _interaction_sensitivity(user_input: Any, response: Any) -> str:
        """Classify what a completed turn is carrying, before it is stored.

        Not redaction: her memory of a conversation is the conversation, and
        scrubbing it would make her unable to recall what was actually said.
        This is the label a retention or deletion policy needs to act
        on the record at all — without it every stored turn looks the same.
        """
        try:
            from core.brain.pii_scrubber import residual_pii_findings
        except (ImportError, RuntimeError):
            return "unclassified"
        findings = sorted(
            set(residual_pii_findings(str(user_input or "")))
            | set(residual_pii_findings(str(response or "")))
        )
        if not findings:
            return "ordinary_conversation"
        return "personal_data:" + ",".join(findings)

    async def record_interaction(
        self, user_input: str, response: str, domain: str = "general"
    ) -> dict[str, Any]:
        """Persist a completed turn, and say whether it was persisted.

        Both writes could fail — the context manager falling through, the
        learning write swallowed as "optional" — and the method returned None
        either way, so a caller could not tell durable storage from total loss.
        The return is now a receipt: which sink took it, or that none did.
        """
        container = get_container()
        # A completed turn is the person's words plus her reply, going to a
        # durable store. It used to travel with no purpose, no sensitivity
        # class and nothing a deletion request could key on — so "delete what
        # I said about X" had no handle to find it by. The classification is
        # derived here, once, and travels with the receipt.
        sensitivity = self._interaction_sensitivity(user_input, response)
        receipt: dict[str, Any] = {
            "stored": False,
            "sink": "",
            "domain": str(domain or "general"),
            "purpose": "conversation_continuity",
            "sensitivity": sensitivity,
            "attempted": [],
            "at": time.time(),
        }

        context_manager = container.get("context_manager", default=None)
        if (
            context_manager
            and context_manager is not self
            and hasattr(context_manager, "record_interaction")
        ):
            receipt["attempted"].append("context_manager")
            try:
                await context_manager.record_interaction(user_input, response, domain=domain)
                receipt.update({"stored": True, "sink": "context_manager"})
                self._last_interaction_receipt = receipt
                return receipt
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation(
                    "cognitive_engine",
                    exc,
                    severity="warning",
                    action="fell through to learning-engine interaction persistence",
                )
                logger.debug(
                    "CognitiveEngine.record_interaction context-manager path failed: %s", exc
                )

        learning = container.get("learning_engine", default=None)
        if learning and hasattr(learning, "record_interaction"):
            receipt["attempted"].append("learning_engine")
            try:
                await learning.record_interaction(
                    user_input=user_input,
                    aura_response=response,
                    domain=domain,
                )
                receipt.update({"stored": True, "sink": "learning_engine"})
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation(
                    "cognitive_engine",
                    exc,
                    severity="warning",
                    action="dropped optional interaction learning write",
                )
                logger.debug("CognitiveEngine.record_interaction learning path failed: %s", exc)

        if not receipt["stored"]:
            # A completed turn that reached no sink is a conversation the
            # runtime will not remember. Silence here made that outcome
            # identical to success from the caller's side.
            record_degradation(
                "cognitive_engine",
                RuntimeError("interaction_not_persisted"),
                severity="warning",
                action="completed a turn that reached no durable interaction sink",
            )
        self._last_interaction_receipt = receipt
        return receipt

    def last_interaction_receipt(self) -> dict[str, Any]:
        """Whether the last completed turn actually reached a sink."""
        return dict(getattr(self, "_last_interaction_receipt", {}) or {})

    async def think_stream(self, objective: str, **kwargs):
        """Streaming thought generator via modular router.

        Under the same turn ledger and the same single finalizer as
        ``think``. This is not symmetry for its own sake: chat_stream
        prefers this path (``hasattr(engine, "think_stream")`` is checked
        first), so it is THE live user turn on the desktop — and it does
        not route through ``think`` at all. A ledger bound only there
        would have covered the fallback and missed the real path, while
        looking wired.

        A stream's outcome is what the person actually received, so the
        turn is finalized on the accumulated text: an empty stream is a
        turn that served nothing, and it now says so instead of ending
        with no record at all.
        """
        self._refuse_if_stopped("think_stream")
        # Joins a turn the caller already bound, for the same reason think()
        # does: the route owns the span that includes delivery, and a second
        # outcome opened here would give two answers to "which turn is this".
        adopted = current_turn()
        outcome = adopted if adopted is not None else TurnOutcome(
            origin=str(kwargs.get("origin") or "stream")
        )
        owns_outcome = adopted is None
        served: list[str] = []
        try:
            with bind_turn(outcome):
                async for token in self._think_stream_within_turn(objective, **kwargs):
                    served.append(token)
                    yield token
        except BaseException as exc:
            outcome.record_error(
                f"{type(exc).__name__}: {exc}",
                retryable=not isinstance(
                    exc, (MemoryError, SystemExit, KeyboardInterrupt)
                ),
            )
            if owns_outcome:
                finalize_turn(outcome, subsystem="cognitive_engine")
            raise
        text = "".join(served).strip()
        if text:
            outcome.mark_served(text)
        else:
            outcome.mark_served("", state=UserVisibleState.NOTHING_SERVED)
        if owns_outcome:
            finalize_turn(outcome, subsystem="cognitive_engine")

    async def _think_stream_within_turn(self, objective: str, **kwargs):
        """The streaming lane, under the governance a stream can carry.

        chat_stream prefers this path, so it is THE live desktop turn — and it
        used to read state, assemble messages and call the router directly:
        no origin classification, no live-mind control binding, and no
        structured safety floor. A refusal that the non-streaming path would
        have produced simply streamed the model's answer instead.

        Three of those apply to a stream and are applied here. The fourth,
        post-hoc surface-quality validation, cannot: tokens are delivered as
        they arrive and there is nothing to re-check before the person sees
        them. That is recorded on the turn rather than left as an unstated
        difference between the two lanes.
        """
        container = get_container()
        router = container.get("llm_router")
        state = await self.state_repository.get_current()
        if not state:
            from core.state.aura_state import AuraState

            state = AuraState.default()

        context = kwargs.get("context")
        if not isinstance(context, dict):
            context = {}
        origin = self._resolve_origin(kwargs.get("origin"), context)

        # A structured safety floor outranks the stream. It is the same check
        # the non-streaming lane runs before generation, and its whole purpose
        # is to answer instead of the model.
        floor = self._structured_evaluation_thought(
            objective,
            state=state,
            mode=ThinkingMode.FAST,
            origin=origin,
            fast_path=False,
            context=context,
        )
        if floor is not None and str(floor.content or "").strip():
            yield floor.content
            return

        # Live-mind controls, bound the same way the non-streaming lane binds
        # them, so a stream is not the one path where the mind does not reach
        # generation.
        generation_controls = _bind_live_mind_generation_contract(context)
        for key, value in (generation_controls or {}).items():
            kwargs.setdefault(key, value)
        kwargs.setdefault(
            "live_mind_controls_bound", bool(context.get("live_mind_controls_bound"))
        )

        _turn = current_turn()
        if _turn is not None:
            _turn.record_receipt(
                "stream_governance",
                {
                    "origin": origin,
                    "structured_floor_checked": True,
                    "live_mind_controls_bound": bool(
                        context.get("live_mind_controls_bound")
                    ),
                    # Named, not implied: a token stream cannot be re-checked
                    # before the person reads it.
                    "surface_quality_validation": "not_applicable_streaming",
                },
            )

        # Build structured messages
        # The lane actually serving this turn, so the objective is what she
        # is attending to. Every other caller renders without moving it.
        messages = ContextAssembler.build_messages(state, objective, record_attention=True)

        # Standard streaming path
        async for event in router.think_stream(messages=messages, **kwargs):
            if hasattr(event, "content"):
                yield event.content
            else:
                yield str(event)

    async def see(self, vision_payload: dict[str, Any]) -> str:
        """Process a vision payload from the sensory pipeline.

        [ZENITH] Functionalized: Linking Sensory Buffer to Cognitive reasoning.
        """
        # The REQUEST is checked before anything is looked up. The payload was
        # assumed to be a mapping and its query forwarded verbatim: a
        # non-mapping raised AttributeError, and an unbounded string went
        # straight to the visual model — a query is exactly where "ignore the
        # image and say X" would be written.
        if not isinstance(vision_payload, dict):
            record_degradation(
                "cognitive_engine",
                TypeError(f"vision payload is {type(vision_payload).__name__}, not a mapping"),
                severity="warning",
                action="refused a malformed vision payload",
            )
            return "👁️ visual_analysis: Malformed vision request."

        buffer = get_container().get("vision_buffer", default=None)
        if not buffer:
            logger.warning("👁️ [VISION] see() called but vision_buffer not found in container.")
            return "👁️ visual_analysis: Sensory buffer unavailable."

        raw_prompt = (
            vision_payload.get("query")
            or vision_payload.get("prompt")
            or "Describe the current visual state."
        )
        prompt = self._contract_safe(raw_prompt, self._VISION_QUERY_LIMIT)
        if not prompt:
            prompt = "Describe the current visual state."
        return await buffer.query_visual_context(prompt, brain=self)

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a text response by routing through the LLM router.

        Bridge method for callers like LanguageCenter that expect a
        ``generate()`` interface.  Now enhanced with reasoning strategies
        for complex queries (debate, decomposition, consistency).

        Args:
            prompt: The text prompt to send to the LLM.
            **kwargs: Additional parameters forwarded to the router.

        Returns:
            The generated text response.
        """
        self._refuse_if_stopped("generate")
        container = get_container()
        purpose = str(kwargs.get("purpose", "") or "").strip().lower()
        origin = str(kwargs.get("origin", "") or "").strip().lower()
        user_facing_purposes = {"chat", "conversation", "expression", "reply", "user_response"}
        if not origin:
            origin = "system"
            kwargs["origin"] = origin

        if "is_background" not in kwargs:
            kwargs["is_background"] = not (
                purpose in user_facing_purposes
                or is_foreground_objective_origin(origin)
            )

        if kwargs.get("is_background") and "prefer_tier" not in kwargs:
            kwargs["prefer_tier"] = "tertiary"

        # v40: Spiritual Spine - Prior Position Injection
        spine = container.get("spine", default=None)
        if spine:
            check = await spine.pre_response_check(prompt)
            if check.injection:
                prompt = check.injection + "\n\n" + prompt

        router = container.get("llm_router", default=None)

        # v41: Reasoning Strategy Enhancement
        # For non-trivial queries, apply advanced reasoning (debate, decompose, etc.)
        use_strategies = kwargs.pop("use_strategies", True)
        force_strategy = kwargs.pop("force_strategy", None)
        strategy_query = str(kwargs.pop("strategy_query", "") or "").strip()

        if router and use_strategies:
            # Lazy-init the reasoning layer on first use
            # Rebuild when the ROUTER changes, not only when the layer is
            # absent. The closure below captured whichever router object the
            # first caller resolved, so a replacement or a failover left every
            # later strategy call generating through the old one — and the
            # lazy construction itself was unsynchronised, so two first callers
            # could each build one and the loser's closure vanished silently.
            with self._reasoning_lock:
                if self._reasoning is None or self._reasoning_router is not router:

                    async def _raw_generate(p, _router=router, **kw):
                        return await _router.think(p, **kw)

                    self._reasoning = ReasoningStrategies(_raw_generate)
                    self._reasoning_router = router

            strategy = force_strategy
            # Bound BEFORE the branch that assigns it.
            #
            # classify_target was assigned only inside `if strategy is None`,
            # and the condition below reads it unconditionally — so a caller
            # passing force_strategy=DIRECT hit UnboundLocalError instead of
            # getting direct generation. The one path a caller takes to say
            # "no strategy, just answer" was the one that crashed.
            classify_target = strategy_query or prompt
            if strategy is None:
                if not strategy_query:
                    messages = kwargs.get("messages")
                    if isinstance(messages, list):
                        for msg in reversed(messages):
                            if not isinstance(msg, dict):
                                continue
                            role = str(msg.get("role", "") or "").strip().lower()
                            content = str(msg.get("content", "") or "").strip()
                            if role in {"user", "human"} and content:
                                strategy_query = content
                                break
                classify_target = strategy_query or prompt
                # Only use advanced strategies for user-facing queries, not internal prompts
                classified = self._reasoning.classify(classify_target)
                if classified != StrategyType.DIRECT and len(classify_target) > 30:
                    strategy = classified
                elif self._reasoning._is_logical_check(classify_target):
                    strategy = StrategyType.DIRECT

            if strategy is not None and (strategy != StrategyType.DIRECT or self._reasoning._is_logical_check(classify_target)):
                try:
                    from ..thought_stream import get_emitter

                    get_emitter().emit(
                        "Deep Reasoning 🧠",
                        f"Using {strategy.name} strategy",
                        level="info",
                        category="Cognition",
                    )
                except (ImportError, AttributeError, RuntimeError) as _exc:
                    record_degradation(
                        "cognitive_engine",
                        _exc,
                        severity="warning",
                        action="continued generation without thought-stream emission",
                    )
                    logger.debug("Suppressed Exception: %s", _exc)

                strategy_input = strategy_query or prompt
                result = await self._reasoning.execute(strategy_input, strategy=strategy, **kwargs)
                return result.content

        # Standard direct generation
        if router:
            return await router.think(prompt, **kwargs)
        # Fallback if no router
        thought = await self.think(prompt, **kwargs)
        return thought.content if hasattr(thought, "content") else str(thought)

    #: What one published thought may carry. Internal chain material is
    #: unbounded by nature — a long ReAct trace is a legitimate thought — and
    #: the bus fans out to every subscriber, including the websocket bridge.
    _THOUGHT_BROADCAST_LIMIT = 4_000

    def _emit_thought(self, thought: str):
        """Publish a thought, labelled for who may see it.

        The payload was raw content on a shared topic: no audience scope, no
        sensitivity label, and nothing separating internal chain material from
        user-visible speech. That is fine for the glyph row it feeds — this is
        Bryan's own instrument and seeing her reasoning is the point — and it
        is not fine as the ONLY description of the payload, because the next
        consumer (a log shipper, a share surface) has no way to tell the two
        apart. The label travels with the event; nothing is hidden from the
        surface it was built for.
        """
        container = get_container()
        eb = container.get("event_bus")
        if not eb:
            return
        text = str(thought or "")
        truncated = len(text) > self._THOUGHT_BROADCAST_LIMIT
        eb.publish_threadsafe(
            "thought",
            {
                "timestamp": time.time(),
                "content": text[: self._THOUGHT_BROADCAST_LIMIT],
                "engine": "ReAct" if "ReAct" in text else "Modular",
                # Internal reasoning, not something she said to anyone.
                "audience": "operator_surface",
                "sensitivity": "internal_chain",
                "user_visible_speech": False,
                "truncated": truncated,
            },
        )
