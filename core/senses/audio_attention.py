"""Evidence-based source attribution and attention policy for heard speech."""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

from core.runtime.lockdep import LockRank, checked_lock

#: How far above the room's own noise floor speech has to sit before it counts
#: as coming from someone AT the machine.
#:
#: LIVE DEFECT, 2026-08-10: "when i talk to my computer nothing happens. no
#: response or anything." The owner's speech was classified `ambient_speech`,
#: which core/senses/voice_engine.py refuses outright — it is not in
#: _SPEAKER_AT_THE_MACHINE_SOURCES — so no utterance could ever be answered.
#:
#: The cause was `near_field = rms_db >= -22.0`: an ABSOLUTE level compared
#: against an uncalibrated microphone. -22 dBFS is a property of one particular
#: input gain, and nothing else. A MacBook's built-in mic at ordinary speaking
#: distance sits below it, so on that hardware the owner is permanently
#: reclassified as distant room noise and the open microphone behaves like a
#: closed one. No phrasing, volume or proximity a person would think to try can
#: fix it, because the number does not describe them — it describes a gain
#: stage.
#:
#: A signal-to-noise margin is a physical property of the situation rather than
#: of the hardware: speech from someone at the keyboard stands well clear of
#: the room it is spoken in, whatever the mic converts that to in dBFS. Every
#: microphone, room and gain setting therefore calibrates itself.
_NEAR_FIELD_SNR_DB = 12.0

#: Recent loudness observations, newest last, used to estimate the room.
#: Bounded because this is a rolling picture of the last few minutes of audio,
#: not a history.
_RMS_HISTORY_MAX = 240

#: Observations required before the estimate is trusted. Below this the
#: absolute threshold still applies, so a fresh process cannot decide the whole
#: room is near-field from a single quiet sample.
_RMS_HISTORY_MIN = 12

#: The fallback absolute threshold, used only while uncalibrated.
_ABSOLUTE_NEAR_FIELD_DB = -22.0

_rms_history: deque[float] = deque(maxlen=_RMS_HISTORY_MAX)
_rms_lock = checked_lock("audio_attention.rms", rank=LockRank.LEAF)


def observe_room_loudness(rms_db: float) -> None:
    """Record one loudness observation for the noise-floor estimate."""
    try:
        value = float(rms_db)
    except (TypeError, ValueError):
        return
    if value != value or value in (float("inf"), float("-inf")):
        return
    with _rms_lock:
        _rms_history.append(value)


def room_noise_floor_db() -> float | None:
    """The quiet end of recent audio, or None while uncalibrated.

    A low percentile rather than a minimum: a single click or a dropout would
    otherwise define the floor forever and every later utterance would clear
    it. A low percentile tracks the room and ignores the outlier.
    """
    with _rms_lock:
        if len(_rms_history) < _RMS_HISTORY_MIN:
            return None
        ordered = sorted(_rms_history)
    index = max(0, int(len(ordered) * 0.20) - 1)
    return ordered[index]


def reset_room_calibration() -> None:
    """Forget the room. For tests and for a changed input device."""
    with _rms_lock:
        _rms_history.clear()

_DIRECT_ADDRESS_RE = re.compile(
    r"\b(?:(?:hey|hi|hello|okay|ok)\s+)?aura\b",
    re.IGNORECASE,
)
_QUESTION_OR_INTEREST_RE = re.compile(
    r"\b(?:why|how|what|idea|discover|research|learn|explain|wonder|news|story)\b",
    re.IGNORECASE,
)
_MEDIA_APP_MARKERS = {
    "chrome",
    "firefox",
    "music",
    "podcasts",
    "quicktime",
    "safari",
    "spotify",
    "tv",
    "vlc",
    "youtube",
}


@dataclass(frozen=True, slots=True)
class AudioAttentionAssessment:
    source: str
    confidence: float
    addressed_to_aura: bool
    response_authorized: bool
    attention_mode: str
    attention_score: float
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def classify_audio_attention(
    text: str,
    *,
    rms_db: float,
    transcript_confidence: float,
    duration_s: float,
    active_app: str = "",
    explicit_command: bool = False,
    visual_context: dict[str, Any] | None = None,
) -> AudioAttentionAssessment:
    """Classify heard speech without treating uncertain audio as user intent.

    The classifier intentionally emits likelihoods rather than identity claims.
    Wake/session logic remains the authority boundary for conversation dispatch.
    """

    normalized = " ".join(str(text or "").split())
    app = str(active_app or "").strip().lower()
    addressed = bool(_DIRECT_ADDRESS_RE.search(normalized))
    media_context = any(marker in app for marker in _MEDIA_APP_MARKERS)
    long_narrative = duration_s >= 7.0 and len(normalized.split()) >= 8
    # Loudness relative to THIS room and THIS microphone, falling back to the
    # absolute level only until enough audio has been heard to know the room.
    # See _NEAR_FIELD_SNR_DB for why an absolute level could not work.
    observe_room_loudness(rms_db)
    noise_floor = room_noise_floor_db()
    if noise_floor is None:
        loud_enough = rms_db >= _ABSOLUTE_NEAR_FIELD_DB
        loudness_reason = "absolute_level_uncalibrated"
    else:
        loud_enough = (rms_db - noise_floor) >= _NEAR_FIELD_SNR_DB
        loudness_reason = "above_room_noise_floor"
    near_field = loud_enough and transcript_confidence >= -0.45
    visual = dict(visual_context or {})
    visual_updated_at = float(visual.get("updated_at", 0.0) or 0.0)
    visual_fresh = bool(
        visual_updated_at > 0.0
        and max(0.0, time.time() - visual_updated_at) <= 6.0
    )
    visible_person = bool(visual_fresh and visual.get("face_present"))
    speaking_likelihood = _clamp(float(visual.get("speaking_likelihood", 0.0) or 0.0))
    visible_speaker = bool(visible_person and speaking_likelihood >= 0.22)

    reasons: list[str] = []
    if explicit_command:
        source = "direct_user"
        confidence = 0.99
        reasons.append("explicit_voice_capture")
    elif addressed:
        source = "direct_address"
        confidence = 0.97
        reasons.append("wake_or_name_address")
    elif visible_speaker:
        source = "nearby_visible_speaker"
        confidence = max(0.72, speaking_likelihood)
        reasons.extend(("fresh_visible_face", "lower_face_motion"))
    elif media_context and long_narrative:
        source = "device_media"
        confidence = 0.84 if visual_fresh and not visible_person else 0.78
        reasons.extend(("media_app_context", "long_narrative_audio"))
    elif near_field:
        source = "nearby_person"
        confidence = 0.68
        reasons.extend((loudness_reason, "speech_confidence"))
    elif long_narrative or rms_db < -25.0:
        source = "ambient_speech"
        confidence = 0.64
        reasons.append("ambient_or_distant_acoustics")
    else:
        source = "unknown_speech"
        confidence = 0.45
        reasons.append("insufficient_source_evidence")

    attention_score = 1.0 if explicit_command or addressed else 0.18
    if near_field:
        attention_score += 0.18
    if visible_speaker:
        attention_score += 0.22
    if _QUESTION_OR_INTEREST_RE.search(normalized):
        attention_score += 0.16
        reasons.append("semantic_interest_signal")
    if source == "device_media":
        attention_score -= 0.08
    attention_score = _clamp(attention_score)

    if addressed or explicit_command:
        attention_mode = "conversation_candidate"
    elif attention_score >= 0.48:
        attention_mode = "attend"
    elif attention_score >= 0.24:
        attention_mode = "observe"
    else:
        attention_mode = "ignore"

    return AudioAttentionAssessment(
        source=source,
        confidence=_clamp(confidence),
        addressed_to_aura=addressed,
        response_authorized=bool(explicit_command),
        attention_mode=attention_mode,
        attention_score=attention_score,
        reasons=tuple(dict.fromkeys(reasons)),
    )
