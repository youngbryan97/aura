"""How she is, said in words for her attention: the readings of her own state that bid for it.

Each is written here and read back here, from one pattern. A reading is what something is like, and nothing can be done that
makes it so, so it is never a goal. 24 November 2026, eleven of the fifty three goals her record held as under way were
readings from weeks before ("Affective state: dread (arousal=0.91)", "Drive alert: social is depleted (95% urgency)"), each
counted as work she owed and offered to her as something to choose.
"""
from __future__ import annotations

import re
import string

__all__ = ["affect_reading", "body_reading", "drive_reading", "is_a_reading_of_her", "surge_reading", "surprise_reading"]

_DRIVE = "Drive alert: {drive} is depleted ({urgency:.0%} urgency)"
_AFFECT = "Affective state: {emotion} (arousal={arousal:.2f})"
_BODY = "Body integrity alert: {integrity:.1f}% (heat={heat:.1f}°)"
_SURPRISE = "Prediction surprise in {what} (err={surprise:.2f})"
_SURGE = "Phenomenal surge: qualia intensity ||q||={norm:.2f}"
#: As it was written until October 2026.
_SURGE_BEFORE = "Phenomenal Surge: High qualia intensity (||q||={norm:.2f})"


def _as_read(pattern: str) -> str:
    """The pattern with each slot open: a number where it is written as one, and words where it is not."""
    return "".join(
        re.escape(before) + ("" if name is None else r"-?\d+(?:\.\d+)?%?" if spec else r".{1,80}?")
        for before, name, spec, _conversion in string.Formatter().parse(pattern)
    )


_READ = re.compile("^(?:" + "|".join(_as_read(p) for p in (_DRIVE, _AFFECT, _BODY, _SURPRISE, _SURGE, _SURGE_BEFORE)) + r")\.?$", re.IGNORECASE)


def drive_reading(drive: str, urgency: float) -> str:
    return _DRIVE.format(drive=drive, urgency=float(urgency))


def affect_reading(emotion: str, arousal: float) -> str:
    return _AFFECT.format(emotion=emotion, arousal=float(arousal))


def body_reading(integrity: float, heat: float) -> str:
    return _BODY.format(integrity=float(integrity), heat=float(heat))


def surprise_reading(what: str, surprise: float) -> str:
    return _SURPRISE.format(what=what, surprise=float(surprise))


def surge_reading(norm: float) -> str:
    return _SURGE.format(norm=float(norm))


def is_a_reading_of_her(words: object) -> bool:
    """Whether these words are one of the readings above, so a state she was in and nothing to be done."""
    return bool(_READ.match(" ".join(str(words or "").split())))
