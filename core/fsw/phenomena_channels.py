"""core/fsw/phenomena_channels.py — declared channels for fourteen dispositions.

The organs these describe live in nine packages and this file imports none of
them, because a channel declaration is metadata: an id, a unit, limits, and
the path of the file that owns it. Keeping it that way is what lets one
dictionary cover organs that are not allowed to import each other, and it is
the same reason the telemetry layer sits below cognition rather than inside it.

Limits are set where the reading means something rather than where a number
looked round. Most of these channels have no limits at all, because most of
these quantities have no bad value — a coherence of 0.4 is not a fault, and a
yellow band on it would train whoever reads the display to ignore the ones
that are faults.

The ones that do carry limits are the states where a mechanism has gone quiet
while continuing to run, which is the failure this whole group of organs is
most prone to:

* a carer going short herself while still allocating
* a signalling channel nobody can read anything from
* someone whose state is no longer mostly their own
* an arbiter abstaining on everything because no outcome ever resolves
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("Aura.FSW.Phenomena")

GROUP = "phenomena"

CHANNEL_COHERENCE = "identity.coherence"
CHANNEL_UNSUPPORTED = "identity.unsupported_declarations"
CHANNEL_TIME_ON_CYCLE = "expression.time_on_cycle_s"
CHANNEL_SYNCHRONY = "expression.synchrony"
CHANNEL_CARE_GINI = "care.gini"
CHANNEL_CARE_OWN_UNMET = "care.own_unmet"
CHANNEL_CARE_DEPLETED = "care.depleted"
CHANNEL_ACCEPTANCE = "receptivity.acceptance_rate"
CHANNEL_REGARD = "receptivity.mean_regard"
CHANNEL_ABSTENTION = "arbitration.abstention_rate"
CHANNEL_AFFECT_LED = "arbitration.affect_led_fraction"
CHANNEL_ASYMMETRY = "position.mean_asymmetry"
CHANNEL_IMPROVEMENT = "craft.improvement_rate"
CHANNEL_NOVELTY_VALUE = "creativity.novelty_value"
CHANNEL_PREMIUM = "reversibility.premium_paid"
CHANNEL_INFORMATIVE = "signalling.informative_fraction"
CHANNEL_CONTINUATION = "reciprocity.mean_continuation"
CHANNEL_AUTONOMY = "empathy.autonomy"
CHANNEL_MERGED = "empathy.merged"
CHANNEL_PLEASURE = "aesthetic.pleasure"
CHANNEL_MARKERS = "conventions.arbitrary_markers"

EVENT_IDENTITY_UNSUPPORTED = "identity.declared_without_practice"
EVENT_CARE_REFUSED = "care.refused_for_floor"
EVENT_CHANNEL_DEAD = "signalling.channel_dead"
EVENT_MERGED = "empathy.state_no_longer_own"
EVENT_LEARNING_LED = "receptivity.taken_to_find_out"

_declared = False


def declare() -> list[str]:
    """Declare every channel and event for this group. Idempotent."""
    global _declared
    if _declared:
        return []
    try:
        from core.fsw.telemetry_dictionary import ChannelType, EventSeverity, channel, event
    except ImportError as exc:  # pragma: no cover — dictionary always present
        logger.debug("telemetry dictionary unavailable: %s", exc)
        return []

    names: list[str] = []
    for spec in (
        dict(
            identifier=0x1701, name=CHANNEL_COHERENCE, unit="fraction",
            description="Kuramoto order parameter over the practices enacting an identity",
            owner="core/identity/constitutive_identity.py",
        ),
        dict(
            identifier=0x1702, name=CHANNEL_UNSUPPORTED, unit="count",
            type=ChannelType.INT,
            description="labels claimed with no practice enacted behind them",
            owner="core/identity/constitutive_identity.py",
            # One is the whole finding. A system describing itself as
            # something it has stopped doing is the case the module exists to
            # be able to notice at all.
            yellow_high=1.0,
        ),
        dict(
            identifier=0x1703, name=CHANNEL_TIME_ON_CYCLE, unit="s",
            description="seconds spent on actions that have no completion state",
            owner="core/embodiment/expressive_dynamics.py",
        ),
        dict(
            identifier=0x1704, name=CHANNEL_SYNCHRONY, unit="fraction",
            description="phase locking with an external rhythm, over recent bouts",
            owner="core/embodiment/expressive_dynamics.py",
        ),
        dict(
            identifier=0x1705, name=CHANNEL_CARE_GINI, unit="fraction",
            description="inequality of the last care allocation across recipients",
            owner="core/ethics/care_allocation.py",
        ),
        dict(
            identifier=0x1706, name=CHANNEL_CARE_OWN_UNMET, unit="budget",
            description="the carer's own need left unmet while allocating to others",
            owner="core/ethics/care_allocation.py",
            # No limit. The units are whatever the caller allocates in, so
            # there is no absolute amount that means anything, and a limit at
            # zero would fire on a healthy system because the comparison is
            # inclusive. The condition lives on its own channel below.
        ),
        dict(
            identifier=0x1715, name=CHANNEL_CARE_DEPLETED, unit="bool",
            type=ChannelType.BOOL,
            description="giving while going short: the arrangement the floor forbids",
            owner="core/ethics/care_allocation.py",
            # The condition rather than the amount, so red means the thing it
            # says and cannot be reached by an idle system.
            red_high=1.0,
        ),
        dict(
            identifier=0x1707, name=CHANNEL_ACCEPTANCE, unit="fraction",
            description="share of offers accepted",
            owner="core/social/receptivity.py",
        ),
        dict(
            identifier=0x1708, name=CHANNEL_REGARD, unit="fraction",
            description="mean posterior that others mean well",
            owner="core/social/receptivity.py",
        ),
        dict(
            identifier=0x1709, name=CHANNEL_ABSTENTION, unit="fraction",
            description="share of arbitrations declined for want of measured skill",
            owner="core/affect/dual_process_arbiter.py",
            # Abstaining is correct early and correct forever only if nothing
            # is resolving. Sustained near one means outcomes are never being
            # fed back, and the arbiter is a pass-through nobody notices.
            yellow_high=0.80, red_high=0.95,
        ),
        dict(
            identifier=0x170A, name=CHANNEL_AFFECT_LED, unit="fraction",
            description="share of domains where the affective channel outweighs the deliberate one",
            owner="core/affect/dual_process_arbiter.py",
        ),
        dict(
            identifier=0x170B, name=CHANNEL_ASYMMETRY, unit="fraction",
            description="mean of prospect less exposure over scored positions",
            owner="core/environment/prospect_refuge.py",
        ),
        dict(
            identifier=0x170C, name=CHANNEL_IMPROVEMENT, unit="quality_per_attempt",
            description="quality gained per attempt on the skill currently being practised",
            owner="core/learning/craft_practice.py",
        ),
        dict(
            identifier=0x170D, name=CHANNEL_NOVELTY_VALUE, unit="fraction",
            description="novelty times intelligibility of the last artifact offered",
            owner="core/creativity/novelty_value.py",
        ),
        dict(
            identifier=0x170E, name=CHANNEL_PREMIUM, unit="cost",
            description="extra cost accepted for options that can be undone",
            owner="core/morality/reversible_alternative.py",
        ),
        dict(
            identifier=0x170F, name=CHANNEL_INFORMATIVE, unit="fraction",
            description="share of received signals a receiver could draw anything from",
            owner="core/social/costly_signaling.py",
            # Zero with signals still going out is a channel that has died
            # while continuing to cost whatever it costs.
            red_low=0.0,
        ),
        dict(
            identifier=0x1710, name=CHANNEL_CONTINUATION, unit="fraction",
            description="mean estimated chance a relationship has another round in it",
            owner="core/social/reciprocity_engine.py",
        ),
        dict(
            identifier=0x1711, name=CHANNEL_AUTONOMY, unit="fraction",
            description="share of her own rest state owed to her own set point",
            owner="core/affect/empathic_coupling.py",
            # Below half, most of where she is came from somebody else. That
            # is the line between being moved and being merged, and it is the
            # one reading in this group worth waking somebody for.
            yellow_low=0.60, red_low=0.50,
        ),
        dict(
            identifier=0x1712, name=CHANNEL_MERGED, unit="count",
            type=ChannelType.INT,
            description="people in the field whose state is mostly not their own",
            owner="core/affect/empathic_coupling.py",
        ),
        dict(
            identifier=0x1713, name=CHANNEL_PLEASURE, unit="hedonic",
            description="Berlyne hedonic value of the last thing looked at",
            owner="core/perception/aesthetic_response.py",
        ),
        dict(
            identifier=0x1714, name=CHANNEL_MARKERS, unit="count",
            type=ChannelType.INT,
            description="markers held as conventional rather than as facts",
            owner="core/social/conventions.py",
        ),
    ):
        try:
            channel(group=GROUP, stale_after_s=900.0, **spec)  # type: ignore[arg-type]
            names.append(str(spec["name"]))
        except (ValueError, KeyError) as exc:
            logger.debug("channel %s not declared: %s", spec.get("name"), exc)

    for spec in (
        dict(
            identifier=0x1780, name=EVENT_IDENTITY_UNSUPPORTED,
            severity=EventSeverity.WARNING_LO,
            format_string="identity {identity} declared as {label} with nothing enacting it",
            description="a label with no supporting practice in the window",
            owner="core/identity/constitutive_identity.py",
        ),
        dict(
            identifier=0x1781, name=EVENT_CARE_REFUSED,
            severity=EventSeverity.ACTIVITY_HI,
            format_string="held back {reserved} for the floor against {need} of need",
            description="the self-floor bound an allocation. Working as intended.",
            owner="core/ethics/care_allocation.py",
        ),
        dict(
            identifier=0x1782, name=EVENT_CHANNEL_DEAD,
            severity=EventSeverity.WARNING_LO,
            format_string="{signals} signals sent, none readable",
            description="effort is being spent on a channel that separates nobody",
            owner="core/social/costly_signaling.py",
        ),
        dict(
            identifier=0x1783, name=EVENT_MERGED,
            severity=EventSeverity.WARNING_HI,
            format_string="{who} is now {share} their own",
            description="a state that is mostly somebody else's",
            owner="core/affect/empathic_coupling.py",
        ),
        dict(
            identifier=0x1784, name=EVENT_LEARNING_LED,
            severity=EventSeverity.DIAGNOSTIC,
            format_string="accepted {label} from {source} for what it would settle",
            description="an offer taken on its information value rather than its worth",
            owner="core/social/receptivity.py",
        ),
    ):
        try:
            event(**spec)  # type: ignore[arg-type]
            names.append(str(spec["name"]))
        except (ValueError, KeyError) as exc:
            logger.debug("event %s not declared: %s", spec.get("name"), exc)

    _declared = True
    return names


def reset_for_test() -> None:
    global _declared
    _declared = False


def channel_names() -> list[str]:
    """Every channel this group owns, declared or not. For the wiring test."""
    return [
        CHANNEL_COHERENCE, CHANNEL_UNSUPPORTED, CHANNEL_TIME_ON_CYCLE,
        CHANNEL_SYNCHRONY, CHANNEL_CARE_GINI, CHANNEL_CARE_OWN_UNMET,
        CHANNEL_CARE_DEPLETED, CHANNEL_ACCEPTANCE, CHANNEL_REGARD,
        CHANNEL_ABSTENTION,
        CHANNEL_AFFECT_LED, CHANNEL_ASYMMETRY, CHANNEL_IMPROVEMENT,
        CHANNEL_NOVELTY_VALUE, CHANNEL_PREMIUM, CHANNEL_INFORMATIVE,
        CHANNEL_CONTINUATION, CHANNEL_AUTONOMY, CHANNEL_MERGED,
        CHANNEL_PLEASURE, CHANNEL_MARKERS,
    ]


def event_names() -> list[str]:
    return [
        EVENT_IDENTITY_UNSUPPORTED, EVENT_CARE_REFUSED, EVENT_CHANNEL_DEAD,
        EVENT_MERGED, EVENT_LEARNING_LED,
    ]


def as_dict() -> dict[str, Any]:
    return {"group": GROUP, "channels": channel_names(), "events": event_names()}
