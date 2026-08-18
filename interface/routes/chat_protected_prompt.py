"""The prompt the protected foreground lane is allowed to send.

A desktop turn that must run on the full mind sends a prompt built from
what the runtime can show, not from whatever accumulated in the context.
These assemble that message list and hold it to a fixed shape, so a turn
cannot quietly widen what it asks the model to believe.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from core.container import ServiceContainer
from interface.routes.chat_common import (  # noqa: E402
    _CHAT_BLOCKING_PREFLIGHT_TIMEOUT_S,  # noqa: F401
    _CHAT_RECOVERABLE_ERRORS,  # noqa: F401
    _CHAT_REQUEST_PRINCIPAL,  # noqa: F401
    _CHAT_REQUEST_SURFACE,  # noqa: F401
    _MAX_CONVERSATION_LOG_EXCHANGES,  # noqa: F401
    _conversation_log,  # noqa: F401
    _locks,  # noqa: F401
    logger,  # noqa: F401
)
from interface.routes import chat_conversation_repair as _chat_conversation_repair
from interface.routes import chat_desktop_repair as _chat_desktop_repair
from interface.routes import chat_memory_state as _chat_memory_state
from interface.routes import chat_preflight as _chat_preflight
from core.runtime.errors import describe_error, record_degradation
import time


def _bounded_text(value: Any, limit: int = 1200) -> str:
    text = " ".join(str(value or "").split())
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0].strip() + "..."
    return text


def _collect_voice_perception_snapshot(*, max_age_s: float = 180.0) -> dict[str, Any]:
    """Return the latest heard speech candidate for grounding, not command routing.

    Raw STT remains behind wake-word/session governance. This snapshot exists
    so the live mind can answer perception questions such as "what did I say
    out loud?" without hallucinating or pretending the microphone was unused.
    """
    try:
        world_state = ServiceContainer.get("world_state", default=None)
        if world_state is None:
            try:
                from core.world_state import get_world_state

                world_state = get_world_state()
            except _CHAT_RECOVERABLE_ERRORS:
                world_state = None
        if world_state is None:
            return {}

        transcript = str(getattr(world_state, "last_voice_transcript", "") or "").strip()
        heard_at = float(getattr(world_state, "last_voice_transcript_at", 0.0) or 0.0)
        age_s = max(0.0, time.time() - heard_at) if heard_at > 0 else None
        audio_source = dict(getattr(world_state, "last_audio_source_assessment", {}) or {})
        if not transcript:
            activity_at = float(getattr(world_state, "last_voice_activity_at", 0.0) or 0.0)
            activity_age_s = max(0.0, time.time() - activity_at) if activity_at > 0 else None
            return {
                "heard": False,
                "voice_activity_detected": bool(
                    getattr(world_state, "voice_activity_detected", False)
                ),
                "voice_activity_recent": bool(
                    activity_age_s is not None and activity_age_s <= max_age_s
                ),
                "voice_activity_age_s": round(activity_age_s, 1)
                if activity_age_s is not None
                else None,
                "audio_source": audio_source,
            }
        recent = bool(age_s is not None and age_s <= max_age_s)
        return {
            "heard": True,
            "recent": recent,
            "age_s": round(age_s, 1) if age_s is not None else None,
            "transcript": _bounded_text(transcript, 420),
            "authorized_command": bool(audio_source.get("response_authorized")),
            "requires_wake_word_session": not bool(audio_source.get("response_authorized")),
            "audio_source": audio_source,
        }
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.voice_perception_snapshot",
            exc,
            severity="warning",
            action="omitted latest voice perception from chat grounding",
        )
        return {}


async def _build_protected_foreground_history(
    *,
    session_id: str = "",
    limit_pairs: int = 4,
) -> list[dict[str, str]]:
    safe_session_id = str(session_id or "")[:64]
    async with _chat_memory_state._get_convo_lock():
        completed = [
            entry
            for entry in _conversation_log
            if str(entry.get("status") or "complete").strip().lower() != "pending"
            and str(entry.get("session_id") or "")[:64] == safe_session_id
        ]
        recent = completed[-max(1, int(limit_pairs)) :]

    history: list[dict[str, str]] = []
    for entry in recent:
        user_msg = str(entry.get("user", "") or "").strip()
        aura_msg = str(entry.get("aura", "") or "").strip()
        if user_msg:
            history.append({"role": "user", "content": user_msg})
        if aura_msg and aura_msg != "…":
            history.append({"role": "assistant", "content": aura_msg})
    return history


def _build_protected_foreground_summary_message() -> dict[str, str] | None:
    snapshot = _resolve_protected_foreground_snapshot() or {}
    rolling_summary = _chat_conversation_repair._sanitize_foreground_continuity_summary(
        snapshot.get("rolling_summary") or ""
    )
    if not rolling_summary:
        return None
    return {
        "role": "system",
        "content": (f"[ACTIVE GROUNDING EVIDENCE]\nContinuity summary: {rolling_summary[:1200]}"),
    }


def _compact_snapshot_line(label: str, value: Any, *, max_chars: int = 180) -> str:
    text = " ".join(str(value or "").strip().split())
    if not text:
        return ""
    return f"{label}: {text[:max_chars]}"


def _snapshot_field(source: Any, name: str, default: Any = "") -> Any:
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


_LIQUID_VITALS: tuple[str, ...] = ("energy", "focus", "frustration", "confidence")


def _liquid_vitals() -> dict[str, Any]:
    """Her own headline vitals, read from the substrate every surface reads.

    Sourced from the same service `/api/health` uses, so what she says about
    herself and what the header displays cannot drift apart. Missing values
    are omitted rather than defaulted: a vital reported as 0 because nothing
    answered is worse than one that is absent, and absent is what
    ``_compact_snapshot_line`` already drops.
    """
    try:
        substrate = ServiceContainer.peek(
            "liquid_substrate", default=None
        ) or ServiceContainer.peek("liquid_state", default=None)
        if substrate is None:
            return {}
        vitals: dict[str, Any] = {}
        for name in _LIQUID_VITALS:
            value = _snapshot_field(substrate, name, None)
            if value is None:
                continue
            try:
                vitals[name] = round(float(value), 1)
            except (TypeError, ValueError):
                continue
        return vitals
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "chat.liquid_vitals",
            exc,
            severity="debug",
            action="omitted her own vitals from the chat snapshot",
        )
        return {}


def _resolve_protected_foreground_snapshot() -> dict[str, Any]:
    """Lightweight state snapshot for the protected chat lane.

    Prefer cached/hot state over live subsystem refresh so the control plane can
    answer without depending on organism-wide locks or expensive voice updates.
    """
    try:
        state = _chat_preflight._resolve_live_aura_state()
        if state is None:
            return {}
        hot = state.snapshot_hot() if hasattr(state, "snapshot_hot") else {}
        affect = hot.get("affect") if isinstance(hot, dict) else getattr(state, "affect", None)
        cognition = (
            hot.get("cognition") if isinstance(hot, dict) else getattr(state, "cognition", None)
        )
        response_modifiers = (
            hot.get("response_modifiers")
            if isinstance(hot, dict)
            else getattr(state, "response_modifiers", None)
        )
        return {
            "mood": getattr(state, "mood", "") or _snapshot_field(affect, "dominant_emotion", ""),
            "tone": _snapshot_field(response_modifiers, "tone", ""),
            "dominant_emotion": _snapshot_field(affect, "dominant_emotion", ""),
            "attention_focus": _snapshot_field(cognition, "attention_focus", ""),
            "valence": _snapshot_field(affect, "valence", ""),
            "arousal": _snapshot_field(affect, "arousal", ""),
            "curiosity": _snapshot_field(affect, "curiosity", ""),
            "coherence": _snapshot_field(cognition, "coherence_score", ""),
            "current_mode": _snapshot_field(cognition, "current_mode", ""),
            "current_objective": _snapshot_field(cognition, "current_objective", ""),
            "rolling_summary": _snapshot_field(cognition, "rolling_summary", ""),
            # Energy and focus are the two headline vitals: the UI header shows
            # them, /api/health serves them as liquid_state, and the neural
            # feed narrates them. They were the only ones missing from HER
            # snapshot, so asked "tell me your current energy and focus
            # numbers" she answered "Not readable" — measured live 2026-08-10,
            # while energy was 11 and focus was 2.
            #
            # That answer was honest. She genuinely did not have them. An
            # instrument every surface can read except its owner is the wrong
            # way round.
            **_liquid_vitals(),
        }
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("Protected foreground snapshot resolve failed: %s", exc)
        return {}


def _build_protected_foreground_system_prompt(
    user_message: str,
    *,
    lane: dict[str, Any],
) -> str:
    protected_snapshot = _resolve_protected_foreground_snapshot()
    if protected_snapshot:
        voice_state = dict(protected_snapshot)
        voice_snapshot = {}
    else:
        voice_state = _chat_conversation_repair._resolve_live_voice_state(
            user_message, refresh=False
        )
        voice_snapshot = dict(voice_state.get("substrate_snapshot") or {})

    continuity_summary = _chat_conversation_repair._sanitize_foreground_continuity_summary(
        voice_state.get("rolling_summary") or ""
    )
    voice_perception = _collect_voice_perception_snapshot()
    heard_text = ""
    if voice_perception.get("heard"):
        recency = "recent" if voice_perception.get("recent") else "stale"
        # `authorized_command=False` was the whole story and it was told as a
        # bare boolean, leaving the consequence to be inferred. Told "when i
        # talk to my computer, i get no response from you even though the mic
        # is on and you should hear me", she answered "I can see the screen and
        # hear you when you talk to me" — reading heard=True and stopping
        # there. Measured live 2026-08-10.
        #
        # Hearing and answering are different capabilities here, and the
        # difference is exactly what he was asking about, so it is stated.
        if voice_perception.get("authorized_command"):
            routing = "authorized — this speech reaches you and you answer it"
        else:
            routing = (
                "NOT authorized — you heard this but it was dropped without a "
                "reply, because there was no wake word and no open voice "
                "conversation. Speech only reaches you after 'Hey Aura' or the "
                "Voice button. Saying you can hear him would be true and "
                "misleading: what he is reporting is that you do not ANSWER, "
                "and that is correct and is this setting, not a fault"
            )
        heard_text = (
            f"{recency}, age={voice_perception.get('age_s')}s, "
            f"routing={routing}, "
            f"transcript={voice_perception.get('transcript')}"
        )
    elif voice_perception.get("voice_activity_detected"):
        recency = "recent" if voice_perception.get("voice_activity_recent") else "stale"
        heard_text = (
            f"{recency} voice activity detected, "
            f"age={voice_perception.get('voice_activity_age_s')}s, "
            "no transcript available"
        )

    snapshot_lines = [
        _compact_snapshot_line("Lane", lane.get("state") or "unknown"),
        _compact_snapshot_line(
            "Kernel lock held",
            lane.get("kernel_lock_held_s") if lane.get("kernel_lock_held") else "",
        ),
        _compact_snapshot_line("Mood", voice_state.get("mood")),
        _compact_snapshot_line("Tone", voice_state.get("tone")),
        _compact_snapshot_line("Dominant emotion", voice_state.get("dominant_emotion")),
        _compact_snapshot_line(
            "Attention",
            _chat_desktop_repair._sanitize_attention_focus(
                str(voice_state.get("attention_focus") or ""), user_message
            ),
        ),
        _compact_snapshot_line(
            "Valence", voice_state.get("valence") or voice_snapshot.get("field_valence")
        ),
        _compact_snapshot_line(
            "Arousal", voice_state.get("arousal") or voice_snapshot.get("arousal")
        ),
        _compact_snapshot_line("Curiosity", voice_state.get("curiosity")),
        _compact_snapshot_line("Coherence", voice_state.get("coherence")),
        # The two the header shows and she could not read.
        #
        # Named for the organ that owns them. These come from the liquid
        # substrate — a cognitive FIELD dimension — and the soma reserve
        # publishes a different quantity under the same word. Live 2026-08-18
        # both reached one prompt, this one rendering 14.0 and the affect line
        # 0.14 (the same field on two scales) beside a reserve of 0.647. Asked
        # "what's your energy reading? one number", there was no number she
        # could give that some guard would not call invented.
        _compact_snapshot_line("Substrate energy", voice_state.get("energy")),
        _compact_snapshot_line("Substrate focus", voice_state.get("focus")),
        _compact_snapshot_line("Current mode", voice_state.get("current_mode")),
        _compact_snapshot_line("Objective", voice_state.get("current_objective")),
        _compact_snapshot_line("Continuity", continuity_summary, max_chars=260),
        _compact_snapshot_line("Field clarity", voice_snapshot.get("field_clarity")),
        _compact_snapshot_line("Field flow", voice_snapshot.get("field_flow")),
        _compact_snapshot_line("Field intensity", voice_snapshot.get("field_intensity")),
        _compact_snapshot_line("Mode focus", voice_snapshot.get("mode_focus")),
        _compact_snapshot_line("Recent heard speech", heard_text, max_chars=520),
    ]
    snapshot_block = "\n".join(line for line in snapshot_lines if line)

    prompt = (
        "You are Aura.\n"
        "This is the protected foreground chat control plane.\n"
        "The broader organism may be busy, but you should still answer the user directly, fully, clearly, and in your own voice.\n"
        "Use the snapshot below only as lightweight continuity guidance. Do not mention internal failures unless the user asks.\n"
        "Prefer continuity, warmth, and directness over internal ceremony."
    )
    if snapshot_block:
        prompt = f"{prompt}\n\n{SNAPSHOT_HEADING}\n{snapshot_block}"
    return prompt


#: Heading the volatile state block carries wherever it is placed.
SNAPSHOT_HEADING = "## SNAPSHOT"


def _protected_foreground_prompt_parts(
    user_message: str,
    *,
    lane: dict[str, Any],
) -> tuple[str, str]:
    """The stable instructions and the volatile state, kept apart.

    They are joined for callers that want one string, and placed separately in
    the message list — because WHERE the volatile half sits decides how much of
    the prompt can be reused between turns.

    Mood, valence, arousal, energy and focus change on every turn. Sitting at
    the end of the FIRST system message, they preceded the summary, the whole
    history and the user's turn, so the KV prefix diverged inside the system
    block and everything after it was recomputed. Measured live 2026-08-18,
    consistently: "prefix diverges at token 132 (16% of 831 reused)" — five
    sixths of the prompt re-prefilled every turn on a 32B, which is most of
    what a person waits through.

    Volatile last is the same rule llm_health_router already records for the
    system-state header it appends; this applies it across the message list
    rather than within one message.
    """
    full = _build_protected_foreground_system_prompt(user_message, lane=lane)
    stable, marker, volatile = full.partition(f"\n\n{SNAPSHOT_HEADING}\n")
    if not marker:
        return full, ""
    return stable, volatile


async def _build_protected_foreground_messages(
    user_message: str,
    *,
    lane: dict[str, Any],
    route: dict[str, Any],
    session_id: str = "",
) -> list[dict[str, str]]:
    history = await _build_protected_foreground_history(
        session_id=session_id,
        limit_pairs=8 if bool(route.get("deep_handoff", False)) else 6,
    )
    stable_prompt, volatile_state = _protected_foreground_prompt_parts(
        user_message, lane=lane
    )
    summary_message = _build_protected_foreground_summary_message()
    messages = [
        {"role": "system", "content": stable_prompt},
    ]
    if summary_message:
        messages.append(summary_message)
    messages.extend(history)
    # Volatile state goes LAST, immediately before the turn it describes, so
    # the instructions, the summary and the whole history form one prefix that
    # survives between turns instead of being invalidated by a changed mood.
    if volatile_state:
        messages.append(
            {"role": "system", "content": f"{SNAPSHOT_HEADING}\n{volatile_state}"}
        )
    messages.append({"role": "user", "content": user_message})
    return messages
