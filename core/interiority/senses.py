"""core/interiority/senses.py — connecting the channels that had nothing in them.

The channel schema declared `timing`, `prosody`, `face` and `posture`
and every one of them arrived absent, so a read on a person ran on two
text channels and correctly refused to be confident. The reason was not
that Aura has no senses. It is that nothing connected them:
``core/senses/interaction_signals.py`` has been producing typing
hesitation, pause-before-submit, voice RMS and steadiness, gaze
direction and head pose the whole time, and the interiority layer was
not reading any of it.

Three rules govern the translation, and each exists because the
alternative is a confident wrong read.

**Stale is absent, not calm.** A modality that has not reported inside
its own freshness window produces no reading at all. Zero would mean
"measured, and quiet", and a system that cannot tell a silent microphone
from a silent person will describe the first as the second.

**Confidence comes from the sense's own account of itself.** The vision
backend declares its method as a Haar cascade with a pupil threshold and
its reliability as a rough attention indicator, so its readings are
capped accordingly rather than entering at the same strength as a
measurement. A sense that says it is rough is believed about that.

**Nothing here decides what a signal means.** These are channel values;
what they are evidence *for* is settled by the loading matrix in
:mod:`core.interiority.other_minds`, read against that person's own
baseline. Deciding meaning here would put a second, hidden model in
front of the first.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping

from core.interiority.evidence import Reading, absent, inferred, measured
from core.interiority.params import ParamKind, declare
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Interiority.Senses")

_FRESHNESS = declare(
    "interiority.senses.freshness_window_s",
    8.0,
    unit="s",
    basis=(
        "How recently a modality must have reported for its reading to count "
        "as being about now. Eight seconds is longer than a turn and shorter "
        "than a pause in a conversation, so a sample from the previous "
        "exchange does not get read as the current one."
    ),
    kind=ParamKind.CALIBRATION,
    sensitivity=(
        "Long and a stale sample is read as the present; short and every "
        "channel is absent during normal conversational gaps."
    ),
    lower=0.5,
    upper=120.0,
    sweep_range=(2.0, 30.0),
    owner="core/interiority/senses.py",
)

_VISION_CEILING = declare(
    "interiority.senses.vision_confidence_ceiling",
    0.45,
    unit="confidence",
    basis=(
        "The vision backend describes itself as a Haar cascade with a pupil "
        "threshold and calls its own output a rough attention indicator. A "
        "sense that says it is rough is believed about that, so its readings "
        "are capped below the point where a single channel could carry a "
        "confident read."
    ),
    kind=ParamKind.DERIVED,
    sensitivity=(
        "Raise it and a rough face detector starts outvoting channels the "
        "person is not managing, which is the wrong way round."
    ),
    owner="core/interiority/senses.py",
)


def _fresh(updated_at: float, now: float) -> bool:
    return updated_at > 0.0 and (now - updated_at) <= _FRESHNESS.value


def _clamp(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return max(0.0, min(1.0, number))


def channels_from(status: Mapping[str, Any], *, now: float | None = None) -> dict[str, Reading]:
    """Translate one interaction-signals snapshot into channel readings."""
    now = time.time() if now is None else now
    out: dict[str, Reading] = {}

    typing = status.get("typing") or {}
    if _fresh(float(typing.get("updated_at", 0.0) or 0.0), now):
        # Timing is the channel people do not manage, because they do not
        # know they are producing it. Hesitation, the pause before hitting
        # send, and correction rate are all of that kind.
        pause = float(typing.get("pause_before_submit_ms", 0.0) or 0.0)
        out["timing"] = measured(
            _clamp(
                0.45 * _clamp(typing.get("hesitation"))
                + 0.35 * min(1.0, pause / 4000.0)
                + 0.20 * _clamp(typing.get("correction_rate"))
            ),
            source="senses:typing",
            confidence=0.8 if typing.get("active") else 0.5,
        )

    voice = status.get("voice") or {}
    if _fresh(float(voice.get("updated_at", 0.0) or 0.0), now):
        # Prosody carries arousal far better than valence, which is what the
        # loading matrix downstream already assumes. Steadiness enters
        # inverted: an unsteady voice is the signal, a steady one is not.
        out["prosody"] = measured(
            _clamp(
                0.40 * _clamp(voice.get("activation"))
                + 0.30 * _clamp(voice.get("stress_cue"))
                + 0.30 * (1.0 - _clamp(voice.get("steadiness"), 0.5))
            ),
            source="senses:voice",
            confidence=_clamp(voice.get("speech_ratio"), 0.5),
        )

    vision = status.get("vision") or {}
    if _fresh(float(vision.get("updated_at", 0.0) or 0.0), now) and vision.get(
        "sample_available"
    ):
        ceiling = _VISION_CEILING.value
        if vision.get("face_present"):
            # Deliberately inferred rather than measured: the backend says
            # its own output is rough, so its readings cannot enter at the
            # strength of a measurement.
            out["face"] = inferred(
                _clamp(vision.get("mouth_motion_score")),
                min(ceiling, _clamp(vision.get("face_area_ratio"), 0.2) + 0.2),
                source="senses:vision:face",
            )
            # Gaze and head pose are a posture reading: slow-changing, and
            # rarely monitored by the person producing them.
            away = str(vision.get("gaze_direction", "unknown")) not in {
                "center", "camera", "forward"
            }
            out["posture"] = inferred(
                _clamp(1.0 - _clamp(vision.get("attention_available"), 0.5))
                if away
                else _clamp(vision.get("attention_available"), 0.5) * 0.5,
                min(ceiling, 0.35),
                source="senses:vision:pose",
            )

    return out


def live_channels(*, now: float | None = None) -> dict[str, Reading]:
    """Read the running interaction-signals engine. Never raises."""
    try:
        from core.container import ServiceContainer

        engine = ServiceContainer.get("interaction_signals", default=None)
        if engine is None or not hasattr(engine, "get_status"):
            return {}
        return channels_from(engine.get_status(), now=now)
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation(
            "interiority.senses", exc, action="sense channels unavailable this turn"
        )
        return {}


def availability(*, now: float | None = None) -> dict[str, Any]:
    """Which channels a sense is currently feeding, and which are absent.

    Absent is the honest report and it is not the same as quiet. A caller
    that cannot tell a silent microphone from a silent person will
    describe the first as the second.
    """
    channels = live_channels(now=now)
    return {
        "carrying": sorted(channels),
        "absent": sorted(
            c for c in ("timing", "prosody", "face", "posture") if c not in channels
        ),
        "confidences": {k: round(v.confidence, 3) for k, v in channels.items()},
    }


__all__ = ["availability", "channels_from", "live_channels"]
