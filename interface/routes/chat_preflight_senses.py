"""What her senses read for a chat turn before she answers: her microphone, for a spoken turn, and her sight when the
person asks something only looking can answer.

Moved out of interface/routes/chat_preflight.py whole, so that module shrinks rather than grows (tools/lint_module_size.py).
"""
from __future__ import annotations

import asyncio  # noqa: F401  (read by the sight step)
import time
from typing import Any

from core.runtime.errors import record_degradation  # noqa: F401  (read by the sight step)
from interface.routes.chat_common import (
    _CHAT_RECOVERABLE_ERRORS,  # noqa: F401
    _INTERNAL_SURFACE_HEARD_AT,
    logger,  # noqa: F401
)


def _heard_at(request: Any) -> float:
    """When this turn's words came through her microphone, or 0.0 for a turn that was not spoken.

    Her voice session sets it in-process. Her server wake lane posts over loopback with the voice surface header, and
    only a loopback post is believed: a header is something any client can send.
    """
    heard_at = float(_INTERNAL_SURFACE_HEARD_AT.get() or 0.0)
    if heard_at > 0.0:
        return heard_at
    headers = getattr(request, "headers", None)
    client = getattr(request, "client", None)
    host = str(getattr(client, "host", "") or "")
    if headers is None or host not in {"127.0.0.1", "::1", "localhost"}:
        return 0.0
    return time.time() if str(headers.get("X-Aura-Surface") or "").strip().lower() == "voice" else 0.0


def _record_the_microphone_reading(words: Any, request: Any = None) -> dict[str, Any] | None:
    """A turn that came through her microphone carries that reading, as a camera read does.

    LIVE 13 November 2026 he asked aloud "can you hear me?" and she answered that her audio input was inactive and
    his text was landing directly with her: the turn's evidence had no microphone channel in it, so the only reading
    she had of her ears was her own guess. A typed turn records nothing here.
    """
    heard_at = _heard_at(request)
    if heard_at <= 0.0:
        return None
    from core.conversation.turn_evidence_custody import (
        record_turn_grounding,
        record_turn_sensory_evidence,
    )
    from core.senses.turn_evidence import (
        build_microphone_turn_evidence,
        sensory_evidence_grounding_block,
    )

    reading = build_microphone_turn_evidence(words, source="voice_governed_turn", heard_at=heard_at)
    record_turn_sensory_evidence(reading)
    record_turn_grounding(sensory_evidence_grounding_block(reading))
    return reading


async def _run_chat_preflight_sight_how_many(_original_user_message, _turn_sensory_evidence, body):
    # Sight. "How many fingers am I holding up" is answerable only by
    # looking, now, at this resolution — the presence lane's thumbnail
    # cannot count fingers and may be seconds old. So the frame is
    # captured for this turn and read by the multimodal model, and
    # what it saw is injected as an observation she then speaks from.
    #
    # It is injected as a *reading*, not as an answer: the vision
    # model's job is to say what is in the image, and hers is to
    # answer the person. A 2B model asked to also be conversational
    # starts hedging in assistant register instead of saying what is
    # in front of it.
    from interface.routes import chat_preflight as _chat_preflight

    try:
        from core.senses.sight_intent import classify as _classify_sight

        # The classifier embeds the turn through the resident encoder.
        # A forward pass on the API server's loop is a forward pass
        # nobody else's request can run beside (2026-09-16: the
        # server was silent for fifty minutes inside this call).
        _sight = await asyncio.to_thread(_classify_sight, _original_user_message)
        if _sight.kind == "look":
            from core.senses.sight import look as _look

            # Asking what is physically present is consent for this one
            # capture. It does not enable ambient vision or alter the
            # persisted privacy setting.
            _seen = await _look(
                _sight.question,
                explicit_user_consent=True,
            )
            from core.conversation.turn_evidence_custody import (
                record_turn_grounding,
                record_turn_sensory_evidence,
            )
            from core.senses.turn_evidence import (
                build_camera_turn_evidence,
                sensory_evidence_grounding_block,
            )

            _turn_sensory_evidence = build_camera_turn_evidence(
                _sight.question,
                ok=bool(_seen.ok),
                observation=_seen.answer,
                cause=_seen.cause,
                detail=_seen.detail,
                observed_at=(
                    _seen.frame.captured_at if _seen.frame is not None else time.time()
                ),
            )
            record_turn_sensory_evidence(_turn_sensory_evidence)
            record_turn_grounding(sensory_evidence_grounding_block(_turn_sensory_evidence))
            if _seen.ok:
                body.message = (
                    "[you just looked through the camera. This is what "
                    f"you can see right now: {_seen.answer}\n"
                    "Answer them from this — it is your own observation, "
                    "so say it as one. Do not describe it as an image or "
                    "a frame, and do not add anything you cannot see.]\n\n"
                    f"{body.message}"
                )
            else:
                body.message = (
                    "[you understood that they asked you to inspect the "
                    "physical scene and you attempted a fresh camera "
                    "observation. It did not complete. The concrete "
                    f"failure was {_seen.cause}: {_seen.detail}. "
                    "You therefore do not know whether another person "
                    "is physically present. Absence of a frame is not "
                    "evidence that nobody is there. Say what you can and "
                    "cannot establish naturally, without reciting sensor "
                    "status fields or pretending you observed the room.]\n\n"
                    f"{body.message}"
                )
        elif _sight.kind in ("camera_on", "camera_off"):
            _camera_state = await _chat_preflight._apply_camera_control(_sight.kind == "camera_on")
            if _camera_state.get("ok"):
                body.message = (
                    f"[you have just switched the camera "
                    f"{'on' if _sight.kind == 'camera_on' else 'off'} yourself, "
                    f"using the {_camera_state.get('mode', 'camera')} path — it is "
                    "done, not pending. Say so briefly the way a person confirms "
                    "an action.]\n\n"
                    f"{body.message}"
                )
            else:
                body.message = (
                    "[the camera control did not complete. Do not claim it did. "
                    f"The concrete failure was: {_camera_state.get('error', 'unknown')}. "
                    "Explain that briefly and retain the user's requested state.]\n\n"
                    f"{body.message}"
                )
    except _CHAT_RECOVERABLE_ERRORS as _sight_exc:
        record_degradation("chat.sight", _sight_exc)
        logger.debug("Chat sight preflight skipped: %s", _sight_exc)
    return _turn_sensory_evidence
