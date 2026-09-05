"""core/interiority/telemetry.py — what the interior reports about itself.

A subsystem that cannot be debugged interactively has to be understood
afterwards from what it reported, which is why this runtime keeps a
dictionary rather than a pile of counters: every value has an id, a
unit, declared limits, and an owner. The interior is exactly such a
subsystem — it runs inside a turn, its state decays, and by the time
anybody asks why an answer came out the way it did, the state that
produced it is gone.

Seven channels, and each is chosen because a specific failure would show
up in it first.

``interiority.faculties_fired``
    How many of the forty-three produced an activation. Zero for a long
    stretch means the appraisal frame is arriving empty and every
    faculty is correctly declining, which looks like calm and is
    blindness.

``interiority.declines``
    The complement. A faculty that declines is doing the right thing;
    all of them declining at once is the same blindness from the other
    side, and the pair separates a quiet interior from a deaf one.

``interiority.tendency_conflict``
    Disagreement among active action readinesses. This is the input
    upheaval reads, and a sustained high value means the objective
    function is unstable rather than that anything is wrong with the
    machinery.

``interiority.transmission_fidelity``
    The fraction of released quanta that crossed. Low fidelity is a
    real state failing to reach its consumers, which is invisible in any
    design where publishing is a function call.

``interiority.worst_tolerance``
    How much gain the most-adapted channel has lost. High means
    something has been shouting long enough that the interior has
    stopped hearing it.

``interiority.hard_constraints``
    Action classes currently removed from the candidate set. A number
    that only ever grows is a system talking itself out of acting.

``interiority.retention_claims``
    Memories held against compaction. Unbounded growth here is hoarding
    with a conscience, which is the failure mode the retention claim's
    own expiry exists to prevent.
"""

from __future__ import annotations

import logging
from typing import Any

from core.fsw.telemetry_dictionary import ChannelType, EventSeverity, channel, event
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Interiority.Telemetry")

_OWNER = "core/interiority/service.py"
_GROUP = "interiority"

FACULTIES_FIRED = channel(
    0x1701, "interiority.faculties_fired", type=ChannelType.INT, unit="faculties",
    description="Faculties that produced an activation on the last appraisal.",
    owner=_OWNER, group=_GROUP, yellow_high=20, red_high=35,
)
DECLINES = channel(
    0x1702, "interiority.declines", type=ChannelType.INT, unit="faculties",
    description=(
        "Faculties that declined for want of evidence. Healthy; all "
        "forty-three at once means the frame is arriving empty."
    ),
    owner=_OWNER, group=_GROUP, yellow_high=42, red_high=43,
)
TENDENCY_CONFLICT = channel(
    0x1703, "interiority.tendency_conflict", type=ChannelType.FLOAT, unit="ratio",
    description=(
        "Normalised entropy over active action readinesses. The input "
        "upheaval reads."
    ),
    owner=_OWNER, group=_GROUP, yellow_high=0.75, red_high=0.92,
)
TRANSMISSION_FIDELITY = channel(
    0x1704, "interiority.transmission_fidelity", type=ChannelType.FLOAT, unit="ratio",
    description="Fraction of released quanta that reached a consumer.",
    owner=_OWNER, group=_GROUP, yellow_low=0.5, red_low=0.25,
)
WORST_TOLERANCE = channel(
    0x1705, "interiority.worst_tolerance", type=ChannelType.FLOAT, unit="ratio",
    description="Gain the most-adapted channel has lost to sustained input.",
    owner=_OWNER, group=_GROUP, yellow_high=0.7, red_high=0.9,
)
HARD_CONSTRAINTS = channel(
    0x1706, "interiority.hard_constraints", type=ChannelType.INT, unit="classes",
    description="Action classes removed from the candidate set before scoring.",
    owner=_OWNER, group=_GROUP, yellow_high=8, red_high=16,
)
RETENTION_CLAIMS = channel(
    0x1707, "interiority.retention_claims", type=ChannelType.INT, unit="memories",
    description="Memories held against compaction by a faculty.",
    owner=_OWNER, group=_GROUP, yellow_high=64, red_high=256,
)

FACULTY_DECLINED_ALL = event(
    0x1710, "interiority.all_faculties_declined",
    severity=EventSeverity.WARNING_HI,
    format_string=(
        "every faculty declined on {event_id}: the appraisal frame carried "
        "nothing any mechanism could read"
    ),
    description=(
        "Not a quiet interior. An empty frame reaching forty-three "
        "mechanisms that each correctly decline looks identical to calm "
        "and is blindness, so it is an event rather than a low reading."
    ),
    owner=_OWNER,
)
STATE_FAILED_TO_CROSS = event(
    0x1711, "interiority.state_failed_to_cross",
    severity=EventSeverity.WARNING_LO,
    format_string="{faculty} activated at {intensity} and no quantum was released",
    description=(
        "A real state that reached no consumer. Expected occasionally — "
        "release is probabilistic by design — and a pattern on one channel "
        "means that channel's terminal is exhausted."
    ),
    owner=_OWNER,
)
CONSTRAINT_HELD = event(
    0x1712, "interiority.constraint_held",
    severity=EventSeverity.ACTIVITY_HI,
    format_string="{faculty} removed {action_class} from the candidate set: {reason}",
    description=(
        "A value acting as a constraint rather than a weight. Recorded "
        "every time, because a constraint nobody can see being applied is "
        "indistinguishable from one that is not there."
    ),
    owner=_OWNER,
)


def publish(state: Any, *, faculties: int, declines: int) -> None:
    """Write the interior's channels. Never raises."""
    try:
        from core.fsw.telemetry_dictionary import emit_event, write

        fidelity = 1.0
        if state.transmitted:
            crossed = sum(1 for v in state.transmitted.values() if v > 0.0)
            fidelity = crossed / len(state.transmitted)

        write("interiority.faculties_fired", faculties)
        write("interiority.declines", declines)
        write("interiority.tendency_conflict", state.tendency_conflict)
        write("interiority.transmission_fidelity", fidelity)
        write("interiority.hard_constraints", len(state.hard_constraints))
        write("interiority.retention_claims", len(state.retention))

        from core.interiority.receptors import get_receptor_bank

        gains = get_receptor_bank().snapshot()["channels"]
        worst = max((c["tolerance"] for c in gains.values()), default=0.0)
        write("interiority.worst_tolerance", worst)

        if declines and faculties == 0:
            emit_event(
                "interiority.all_faculties_declined",
                event_id=getattr(state, "event_id", "unknown"),
            )
        for faculty in state.failed_to_cross:
            emit_event(
                "interiority.state_failed_to_cross",
                faculty=faculty,
                intensity=round(state.transmitted.get(faculty, 0.0), 4),
            )
        for constraint in state.hard_constraints:
            emit_event(
                "interiority.constraint_held",
                faculty=constraint.held_by,
                action_class=constraint.action_class,
                reason=constraint.reason[:120],
            )
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError, KeyError) as exc:
        record_degradation(
            "interiority.telemetry", exc, action="interior channels not written"
        )


__all__ = [
    "CONSTRAINT_HELD",
    "DECLINES",
    "FACULTIES_FIRED",
    "FACULTY_DECLINED_ALL",
    "HARD_CONSTRAINTS",
    "RETENTION_CLAIMS",
    "STATE_FAILED_TO_CROSS",
    "TENDENCY_CONFLICT",
    "TRANSMISSION_FIDELITY",
    "WORST_TOLERANCE",
    "publish",
]
