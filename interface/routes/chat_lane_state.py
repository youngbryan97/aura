"""The conversation lane: what state it is in, and what that state permits.

A lane is the one place a turn can be in flight, so reading it wrong is how a
turn waits behind an owner that has already gone, or drops to the orchestrator
while the cortex is still loading and would have answered in a moment. The
sentences a person sees when the lane cannot serve them are here too, each one
written to continue a mood prefix rather than start over the top of it.
"""
from __future__ import annotations

import time
from typing import Any

from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from interface.routes import chat_preflight as _chat_preflight  # noqa: E402
from interface.routes.chat_common import (
    _CHAT_RECOVERABLE_ERRORS,
    logger,
)


def _host_condition() -> dict[str, Any]:
    """The machine's own load, as the runtime already measures it."""

    try:
        from core.introspection.self_evidence import resolve_self_health

        readings = {
            reading.channel: reading for reading in resolve_self_health().readings
        }
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.host_condition",
            exc,
            severity="debug",
            action="left the machine's load out of her state snapshot",
        )
        return {}
    condition: dict[str, Any] = {}
    load = readings.get("host_load")
    if load is not None and load.present:
        values = dict(load.value or {})
        for name in ("processor_percent", "memory_percent"):
            try:
                condition[name] = round(float(values.get(name)), 1)
            except (TypeError, ValueError):
                continue
    thermal = readings.get("host_thermal")
    if thermal is not None and thermal.present:
        try:
            condition["thermal_pressure"] = round(float(thermal.value), 2)
        except (TypeError, ValueError):
            pass
    # Absent rather than zero: a load reported as 0% because nothing answered
    # is worse than one that is missing, and she can say she does not know.
    return condition


def _canonical_runtime_model_label(lane: dict[str, Any] | None) -> str:
    lane = dict(lane or {})
    candidates = [
        str(lane.get("desired_model") or ""),
        str(lane.get("last_user_generation_endpoint") or ""),
        str(lane.get("foreground_endpoint") or ""),
        str(lane.get("desired_endpoint") or ""),
        str(lane.get("model_path") or ""),
    ]
    joined = " ".join(candidates).lower()
    # Which LANE this is stays a name match -- the lane names are the words in
    # these fields. What the lane is CALLED comes from the registry, which
    # holds the signed resident descriptor. Reading the size out of these
    # strings could not match a 27B at all, and the "cortex" token was wired
    # to a literal "Cortex (32B)", so the label named a checkpoint that had
    # been replaced while its descriptor sat one call away.
    try:
        from core.brain.llm.model_registry import (
            BRAINSTEM_ENDPOINT,
            DEEP_ENDPOINT,
            FALLBACK_ENDPOINT,
            PRIMARY_ENDPOINT,
            lane_display_label,
        )
    except ImportError:
        lane_display_label = None
    if lane_display_label is not None:
        if "solver" in joined:
            return lane_display_label(DEEP_ENDPOINT)
        if "brainstem" in joined:
            return lane_display_label(BRAINSTEM_ENDPOINT)
        if "reflex" in joined:
            return lane_display_label(FALLBACK_ENDPOINT)
        if "cortex" in joined:
            return lane_display_label(PRIMARY_ENDPOINT)
    if lane.get("desired_model") or lane.get("foreground_endpoint"):
        return str(lane.get("desired_model") or lane.get("foreground_endpoint"))
    if lane_display_label is not None:
        return lane_display_label(PRIMARY_ENDPOINT)
    return "Cortex"


def _conversation_lane_is_standby(lane: dict[str, Any] | None) -> bool:
    lane = dict(lane or {})
    state = str(lane.get("state", "") or "").strip().lower()
    return (
        not bool(lane.get("conversation_ready", False))
        and state in {"cold", "closed", ""}
        and not bool(lane.get("warmup_attempted", False))
        and not bool(lane.get("warmup_in_flight", False))
    )


def _mark_conversation_lane_timeout(reason: str = "foreground_timeout") -> dict[str, Any]:
    from core.brain.llm.model_registry import PRIMARY_ENDPOINT

    # Activate recovery cooldown so rapid follow-up messages are fast-rejected
    # instead of piling into the inference pipeline.
    _enter_recovery_cooldown()
    _force_clear_mlx_foreground_owner(reason=reason, min_age_s=45.0)

    try:
        gate = ServiceContainer.get("inference_gate", default=None)
        if gate and hasattr(gate, "note_foreground_timeout"):
            gate.note_foreground_timeout(reason)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Conversation lane timeout mark failed: %s", exc)

    lane = _chat_preflight._collect_conversation_lane_status()
    lane["state"] = "recovering"
    lane["conversation_ready"] = False
    lane["last_failure_reason"] = reason
    if not lane.get("foreground_endpoint"):
        lane["foreground_endpoint"] = PRIMARY_ENDPOINT
    return lane


def _force_clear_mlx_foreground_owner(
    *,
    reason: str,
    min_age_s: float = 45.0,
) -> dict[str, Any]:
    try:
        from core.brain.llm.mlx_client import force_clear_foreground_owner

        result = force_clear_foreground_owner(
            reason=reason,
            min_age_s=min_age_s,
        )
        if result.get("cleared"):
            logger.warning(
                "Cleared stale MLX foreground owner during chat recovery: %s",
                result,
            )
        return result
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("MLX foreground owner recovery hook unavailable: %s", exc)
        return {
            "cleared": False,
            "reason": reason,
            "holder": None,
            "age_s": 0.0,
            "detail": "unavailable",
        }


def _mark_conversation_lane_state(reason: str, *, state: str) -> dict[str, Any]:
    from core.brain.llm.model_registry import PRIMARY_ENDPOINT

    lane = _chat_preflight._collect_conversation_lane_status()
    lane["state"] = state
    lane["conversation_ready"] = False
    lane["last_failure_reason"] = reason
    lane["warmup_attempted"] = True
    if not lane.get("foreground_endpoint"):
        lane["foreground_endpoint"] = PRIMARY_ENDPOINT
    return lane


def _status_represents_memory_state_result(status: str | None) -> bool:
    return str(status or "").strip() in {
        "owner_identity_recall",
        "session_memory_pin",
        "session_memory_pin_transient",
        "session_memory_recall",
        "session_memory_context_recall",
        "conversation_recall",
    }


def _turn_count_ordinal(count: int) -> str:
    value = max(0, int(count))
    suffix = "th"
    if not 10 <= value % 100 <= 20:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def _cortex_is_cold_loading(lane: object) -> bool:
    """True while the cortex is doing its one-time load for this process.

    The admission budget is the turn's remaining time minus a reserve, and the
    reserve normally includes 60s held back for producing an answer. During a
    COLD load that subtraction is backwards: it deducts time for answering from
    the time needed to become able to answer at all.

    LIVE 2026-08-17, measured four times: the first message after launch died
    at 15s, 15s, 16s and 16s. The foreground timeout is ~80s and the reserve is
    64s, so admission got ~16s — while a cortex cold load needs well over a
    minute. The turn could not have succeeded at any point during boot, and the
    person got "the live answer lane could not finish preparing", which reads
    as a fault rather than as a model still loading.

    A cold load is a one-time wait a person who just launched an app expects to
    pay. There is no answer to reserve for until the weights are up, so during
    that window only the response reserve is held back.
    """

    if not isinstance(lane, dict):
        return False
    try:
        if bool(lane.get("conversation_ready")):
            return False
        if bool(lane.get("has_generated_successfully")):
            return False  # served once already: this is a recovery, not a cold load
        return float(lane.get("last_ready_at") or 0.0) <= 0.0
    except (AttributeError, TypeError, ValueError):
        return False


def _with_the_same_readings(system_prompt: str, readings: list[str]) -> str:
    """The identity, with whatever was read put in front of the model."""
    if not readings:
        return system_prompt
    return "\n\n".join([system_prompt, *readings] if system_prompt else readings)


def _lane_reply_confidence(served: object, default: str) -> str:
    """How much to trust a reply the degraded path produced.

    A deterministic result is the most reliable answer the runtime can give —
    no model, no sampling, no lane. Live 2026-08-19 the exact product 50,420,273
    was served and badged "No answer", which is the opposite of true and
    exactly the kind of thing a person checking her work would catch.
    """
    body = str(served or "").strip()
    known = _known_answer_for_this_turn()
    return "computed" if known and body == known else str(default)


def _lane_status_message_body(
    lane: dict[str, Any],
    *,
    timed_out: bool = False,
    status_override: str = "",
) -> str:
    """Generate a personality-infused status message instead of a robotic error.

    [STABILITY v50] These messages now sound like Aura experiencing a
    momentary lapse rather than a system displaying error codes. Uses
    the live expression frame when available so Aura's current mood
    colours even her recovery messages.
    """
    state = str(lane.get("state", "warming") or "warming")
    failure_reason = str(lane.get("last_failure_reason", "") or "")
    status_override = str(status_override or "")

    # Hard infrastructure failures — keep these explicit for debugging
    if failure_reason.startswith(("mlx_runtime_unavailable:", "local_runtime_unavailable:")):
        model_label = _canonical_runtime_model_label(lane)
        return (
            f"The local {model_label} runtime could not start cleanly. I should not "
            "fake a normal answer; the launcher logs have the failure details."
        )
    if (
        "memory_pressure_refused_worker_spawn" in failure_reason
        or "projected_process_tree_rss" in failure_reason
        or "model_load_headroom" in failure_reason
    ):
        return (
            "The local model lane was blocked by the unified-memory guard before loading. "
            "I am protecting the desktop from an unsafe RAM spike instead of pretending Cortex is merely warming."
        )

    # Build a mood-aware prefix for softer messages
    #
    # Every line below is written to continue one: "Mmm, that answer took too
    # long". With no mood to prefix, the sentence began lowercase and reached
    # the person as a fragment — "that answer took too long to finish
    # cleanly." LIVE 2026-08-26.
    _mood_prefix = ""
    try:
        _pe = ServiceContainer.peek("personality_engine", default=None)
        if _pe and hasattr(_pe, "get_emotional_context_for_response"):
            _emo = _pe.get_emotional_context_for_response() or {}
            _mood = str(_emo.get("mood", "") or "").lower()
            if _mood in {"frustrated", "irritated", "tense"}:
                _mood_prefix = "Ugh, "
            elif _mood in {"tired", "drowsy", "low"}:
                _mood_prefix = "Mmm, "
            elif _mood in {"curious", "playful", "amused"}:
                _mood_prefix = "Hmm — "
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Mood prefix unavailable for degraded reply: %s", exc)

    if status_override == "warming_timeout":
        return (
            _with_mood(_mood_prefix, "the live answer lane exceeded its warm-up budget before "
            "a reasoning turn began. I did not misclassify that boot delay as a failed answer.")
        )
    if status_override == "warming_failed":
        return (
            _with_mood(_mood_prefix, "the live answer lane could not finish preparing before "
            "a reasoning turn began. I recorded the readiness failure separately from Aura's answer quality.")
        )
    if timed_out:
        return _with_mood(_mood_prefix, "that answer took too long to finish cleanly. I logged the timeout and preserved the turn context.")
    if _conversation_lane_is_standby(lane):
        return _with_mood(_mood_prefix, "the local answer path is still preparing. I logged the cold lane instead of claiming Aura is ready.")
    if state == "recovering":
        return _with_mood(_mood_prefix, "the answer lane is recovering from the previous failure. I logged the degraded state instead of emitting a fragment.")
    if state == "failed":
        return _with_mood(_mood_prefix, "the local answer path failed before producing a coherent reply. I'm restarting it instead of pretending that was a real answer.")
    return _with_mood(_mood_prefix, "the answer path is not ready yet; the readiness state is recorded on the live lane.")


def _enter_recovery_cooldown() -> None:
    global _last_recovery_cooldown_at
    _last_recovery_cooldown_at = time.monotonic()


def _conversation_lane_blocks_fallback(lane: dict[str, Any]) -> bool:
    """Avoid hiding a hard local backend failure behind a generic fallback reply."""
    state = str(lane.get("state", "") or "").strip().lower()
    failure_reason = str(lane.get("last_failure_reason", "") or "")
    if state != "failed":
        return False
    return failure_reason.startswith(("mlx_runtime_unavailable:", "local_runtime_unavailable:"))


def _conversation_lane_needs_instant_social_contract(lane: dict[str, Any]) -> bool:
    """Return whether a low-risk presence turn should avoid cold-warming Cortex."""

    state = str(lane.get("state", "") or "").strip().lower()
    if state in {"cold", "warming", "recovering", "failed", "unavailable"}:
        return True
    if lane.get("conversation_ready") is False:
        return True
    blockers = lane.get("readiness_blockers") or ()
    if isinstance(blockers, (list, tuple, set)) and blockers:
        return True
    if not str(lane.get("foreground_endpoint", "") or "").strip() and state not in {
        "ready",
        "healthy",
    }:
        return True
    return False


def _with_mood(prefix: str, sentence: str) -> str:
    """A reply written to follow a mood prefix, said with or without one.

    Each of these lines continues something: "Mmm, that answer took too long".
    With no mood to continue from, the sentence has to start for itself —
    otherwise it reaches the person as a fragment, which is how "that answer
    took too long to finish cleanly." was read out lowercase. LIVE 2026-08-26.
    """
    said = str(sentence or "")
    lead = str(prefix or "")
    if lead:
        return f"{lead}{said}"
    return said[:1].upper() + said[1:] if said else said


def _known_answer_for_this_turn() -> str:
    """What the runtime can answer without the model, or empty.

    A lane that is warming, timed out or recovering says nothing about whether
    the answer is knowable. "what is 7919 * 6367?" has one exact answer, held
    by a deterministic form that needs no generation at all, and it was
    replaced by a sentence about the lane.
    """
    try:
        from core.conversation.arithmetic_check import requested_arithmetic_result
        from core.conversation.session_scope import current_user_question

        question = current_user_question()
        if not question:
            return ""

        # A seating problem the enumeration settles.
        #
        # LIVE, 2026-08-21: the solver produced the answer at 22:42:45 —
        # "took 1 reading(s): the seating, worked out" — and the turn then
        # spent another 105 seconds generating text that was replaced by that
        # same answer at the end. An exact answer is not an improvement on a
        # generated one, it is a reason not to generate.
        from core.reasoning.positional_constraints import (
            answer_positional_problem,
            describe_positional_answer,
        )

        seating = describe_positional_answer(answer_positional_problem(question))
        if seating:
            return seating

        # A game the preflight already enumerated. Worked out before anything
        # was generated, so there is nothing here to improve on.
        #
        # A repository diagnosis is different: it is an observation, and what
        # was asked for was an explanation of it. That one is composed with
        # the reply instead of replacing it, further down.
        from core.conversation.session_scope import solved_answers

        solved = solved_answers()
        settled = solved.get("finite_game", "")
        if settled:
            return settled
        # Anything else the runtime worked out this turn.
        #
        # This function is asked twice: before generating, where only a
        # preflight result exists, and again at the point of giving up, where
        # the comment beside the apology says to ask whether the runtime
        # already HOLDS the answer. A diagnosis is produced by a tool during
        # generation, so it can never skip generation — it can only stop the
        # turn ending in an apology while the finding sits in hand.
        for value in reversed(list(solved.values())):
            if value.strip():
                return value

        value = requested_arithmetic_result(question)
        if value is None:
            return ""
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        shown = f"{value:,}" if isinstance(value, int) else f"{value:,}"
        return f"{shown}."
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.known_answer",
            exc,
            severity="debug",
            action="served the lane status without checking for a computed answer",
            enforce_failure_policy=False,
        )
        return ""
