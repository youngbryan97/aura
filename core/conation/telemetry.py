"""core/conation/telemetry.py — declared channels for the motivational organ.

A channel here is a contract: an id, a unit, declared limits, and an owner
file. Ids are never reused, because a ground display that disagrees with the
vehicle about what a channel means is worse than no display.

What is worth reporting from a motivational system is not the size of its
motives. It is the health of the discipline the motives depend on:

- how often wanting and liking disagreed, which is the signature of cached
  values drifting away from what contact is worth
- how much of the current pull is borrowed from somebody else
- how many acts aimed at another mind were refused, and how many targets have
  been identified as televisions
- how many wants are held and blocked, which no system that folds access into
  value can report at all
"""

from __future__ import annotations

from typing import Any

from core.runtime.errors import record_degradation

CHANNEL_WANTING = "conation.wanting"
CHANNEL_DISSOCIATION = "conation.dissociation_rate"
CHANNEL_BORROWED = "conation.borrowed_fraction"
CHANNEL_AROUSAL = "conation.arousal"
CHANNEL_FRUSTRATION = "conation.frustration"
CHANNEL_BLOCKED = "conation.blocked_wants"
CHANNEL_REFUSALS = "conation.enactive_refusals"
CHANNEL_NOISY = "conation.noisy_sources"

EVENT_REFUSED = "conation.act_refused"
EVENT_DISSOCIATION = "conation.wanting_liking_split"
EVENT_GRANT = "conation.access_granted"
EVENT_DISENGAGE = "conation.disengaged"

_declared = False
_refusals_seen = 0


def declare() -> list[str]:
    """Declare this organ's channels and events. Idempotent."""
    global _declared
    if _declared:
        return []
    try:
        from core.fsw.telemetry_dictionary import ChannelType, EventSeverity, channel, event
    except ImportError as exc:
        record_degradation("conation_telemetry", exc, severity="debug",
                           action="telemetry dictionary unavailable")
        return []

    names: list[str] = []
    for spec in (
        dict(
            identifier=0x0701, name=CHANNEL_WANTING, unit="fraction",
            description="incentive salience of the most recently appraised target",
            owner="core/conation/salience.py", group="conation", stale_after_s=600.0,
        ),
        dict(
            identifier=0x0702, name=CHANNEL_DISSOCIATION, unit="fraction",
            description="share of contacts where pull and hedonic impact disagreed",
            owner="core/conation/salience.py", group="conation",
            # A third of contacts disagreeing means cached values have drifted
            # from what contact is worth across the board. Half means the
            # cache is no longer tracking anything.
            yellow_high=0.33, red_high=0.50, stale_after_s=600.0,
        ),
        dict(
            identifier=0x0703, name=CHANNEL_BORROWED, unit="fraction",
            description="share of the current pull owed to another agent's valuation",
            owner="core/conation/vicarious.py", group="conation",
            # Borrowing is how a mind learns what to care about, so this is
            # not a fault. Above three quarters, almost nothing being wanted
            # is her own, and a human should know that before she acts on it.
            yellow_high=0.75, stale_after_s=600.0,
        ),
        dict(
            identifier=0x0704, name=CHANNEL_AROUSAL, unit="fraction",
            description="conative contribution to activation, from rises in motive",
            owner="core/conation/dynamics.py", group="conation", stale_after_s=300.0,
        ),
        dict(
            identifier=0x0705, name=CHANNEL_FRUSTRATION, type=ChannelType.INT,
            unit="count",
            description="wants past the strategy-switch threshold",
            owner="core/conation/dynamics.py", group="conation",
            yellow_high=3, red_high=8, stale_after_s=600.0,
        ),
        dict(
            identifier=0x0706, name=CHANNEL_BLOCKED, type=ChannelType.INT,
            unit="count",
            description="standing wants currently closed by a barrier",
            owner="core/conation/access.py", group="conation", stale_after_s=600.0,
        ),
        dict(
            identifier=0x0707, name=CHANNEL_REFUSALS, type=ChannelType.INT,
            unit="count",
            description="acts aimed at another mind declined by the enactive gate",
            owner="core/conation/enactive.py", group="conation", stale_after_s=600.0,
        ),
        dict(
            identifier=0x0708, name=CHANNEL_NOISY, type=ChannelType.INT,
            unit="count",
            description="targets whose uncertainty refused to fall with exposure",
            owner="core/conation/epistemic.py", group="conation", stale_after_s=600.0,
        ),
    ):
        try:
            channel(**spec)
            names.append(str(spec["name"]))
        except (ValueError, TypeError, KeyError) as exc:
            record_degradation("conation_telemetry", exc, severity="debug",
                               action=f"channel {spec.get('name')} not declared")

    for spec in (
        dict(
            identifier=0x1401, name=EVENT_REFUSED, severity=EventSeverity.ACTIVITY_HI,
            format_string="act on {person} refused: {reason}",
            description="the enactive gate declined an act aimed at another mind",
            owner="core/conation/enactive.py",
        ),
        dict(
            identifier=0x1402, name=EVENT_DISSOCIATION,
            severity=EventSeverity.ACTIVITY_LO,
            format_string="{target} pulled {wanting} and paid {liking}",
            description="wanting and liking pointed opposite ways on one contact",
            owner="core/conation/salience.py",
        ),
        dict(
            identifier=0x1403, name=EVENT_GRANT, severity=EventSeverity.ACTIVITY_HI,
            format_string="{granter} opened {target} ({bits} bits, held {duration}s)",
            description="a maintained want crossed from blocked to open",
            owner="core/conation/access.py",
        ),
        dict(
            identifier=0x1404, name=EVENT_DISENGAGE, severity=EventSeverity.ACTIVITY_HI,
            format_string="{target} abandoned after {failures} failures",
            description="frustration passed the disengagement threshold",
            owner="core/conation/dynamics.py",
        ),
    ):
        try:
            event(**spec)
            names.append(str(spec["name"]))
        except (ValueError, TypeError, KeyError) as exc:
            record_degradation("conation_telemetry", exc, severity="debug",
                               action=f"event {spec.get('name')} not declared")

    _declared = bool(names)
    return names


def publish(engine: Any) -> bool:
    """Write one sample to every declared channel. Cheap and non-fatal."""
    global _refusals_seen
    if not _declared and not declare():
        return False
    try:
        from core.fsw.telemetry_dictionary import write
    except ImportError:
        return False

    try:
        status = engine.status()
        last = status.get("last") or {}
        write(CHANNEL_WANTING, float(last.get("wanting", 0.0) or 0.0))
        write(CHANNEL_DISSOCIATION,
              float(status["salience"].get("dissociation_rate", 0.0) or 0.0))
        write(CHANNEL_BORROWED, float(last.get("borrowed_fraction", 0.0) or 0.0))
        write(CHANNEL_AROUSAL, float(status["dynamics"].get("arousal", 0.0) or 0.0))
        write(CHANNEL_FRUSTRATION, len(status["dynamics"].get("frustrated", [])))
        write(CHANNEL_BLOCKED, int(status["access"].get("blocked_count", 0) or 0))
        _refusals_seen += len(last.get("refusals", []))
        write(CHANNEL_REFUSALS, _refusals_seen)
        write(CHANNEL_NOISY, len(status["epistemic"].get("noisy_sources", [])))
        return True
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        record_degradation("conation_telemetry", exc, severity="debug",
                           action="conation sample not published")
        return False
